from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from clients.client import Client
from utils.kmeans import TorchKMeans


class CADFedFilterClient(Client):
    """
    Centroid-Guided Client in Asynchronous Decentralized FL
    版本：A — 质心空间 FedProx 正则，用于缓解 non-IID
    (Optimized with Vectorization and CUDA foreach kernels)
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        hp = dict(hyperparam)
        super().__init__(client_id, dataset_index, full_dataset, hp, device)

        # ========== 质心压缩 / 剪枝相关 ==========
        self.cluster_model: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None

        self.mask: Dict[str, torch.Tensor] = {}
        # 优化：缓存参数列表和Mask列表，用于 _foreach 快速计算
        self._param_list_for_masking: List[nn.Parameter] = []
        self._mask_list_for_masking: List[torch.Tensor] = []

        self.n_clusters: int = int(hp.get("n_clusters", 16))
        self.epochs: int = int(hp.get("epochs", 1))
        self.sim_eps: float = float(hp.get("sim_eps", 1e-8))
        self.apply_mask_every_epoch: bool = bool(hp.get("apply_mask_every_epoch", True))

        self.ps_mass: float = 1.0

        # ========== 全局质心 ==========
        self.use_global_cents: bool = bool(hp.get("use_global_cents", True))
        self.global_cents: Dict[str, torch.Tensor] = {}
        self.centroid_reg_lambda: float = float(hp.get("centroid_reg_lambda", 0.0))

    # ============================================================
    # KMeans 聚类辅助
    # ============================================================
    @staticmethod
    def _sort_centroids_and_remap(c1d: torch.Tensor, lbl: torch.Tensor):
        c = c1d.view(-1)  # 避免 detach，保持计算图（如果需要的话，这里不需要）
        order = torch.argsort(c)
        sorted_c = c[order].view(-1, 1)

        # 优化：利用 searchsorted 或者 scatter 可能更快，但此处argsort足够快且稳定
        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        new_labels = old2new[lbl]

        return sorted_c, new_labels

    @torch.no_grad()
    def _cluster_and_prune_model_weights(self):
        clustered: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}
        cents: Dict[str, torch.Tensor] = {}
        labels: Dict[str, torch.Tensor] = {}

        state = self.model.state_dict()
        device = self.device

        # 临时列表用于构建缓存
        param_list = []
        mask_list = []

        for key, w in state.items():
            # 筛选条件保持不变
            if "weight" in key and "bn" not in key and "downsample" not in key:
                orig_shape = w.shape
                flat = w.view(-1, 1)  # view比detach().view()稍微省一点开销，虽然此处no_grad无所谓

                # KMeans 仍然是逐层计算，因每层形状不同难以batch化
                # 注意：如果 TorchKMeans 支持 batch 处理不同分布，可以进一步优化，但通常不支持
                kmeans = TorchKMeans(
                    n_clusters=self.n_clusters,
                    is_sparse=True,
                    init_centroids=None,
                )

                if self.use_global_cents:
                    g = self.global_cents.get(key, None)
                    if g is not None:
                        g_init = g.view(-1, 1)
                        if g_init.shape[0] == self.n_clusters:
                            kmeans.init_centroids = g_init.to(flat.device)

                kmeans.fit(flat)

                cent_sorted, lab_sorted = self._sort_centroids_and_remap(
                    kmeans.centroids, kmeans.labels_
                )

                # 利用 index_select 或者 gather 重构
                # cent_sorted: [K, 1], lab_sorted: [N]
                new_w = torch.index_select(cent_sorted, 0, lab_sorted.view(-1)).view(orig_shape)

                # 剪枝 mask 计算：利用 torch.isin 或者简单的索引
                # 找出值为0的质心索引
                zero_indices = torch.nonzero(cent_sorted.view(-1) == 0, as_tuple=True)[0]

                if zero_indices.numel() > 0:
                    # 如果有0质心，计算 mask
                    # 优化：不直接生成 bool tensor，生成 float mask 以便后面用 mul_
                    # eq(0) 返回 True (需要剪枝)，Mask 应为 0。所以是 ~eq(0)
                    # 或者更简单：new_w != 0
                    m = (new_w != 0).to(dtype=w.dtype)  # 保持和 param 相同类型以便 _foreach_mul
                else:
                    m = torch.ones_like(w)

                clustered[key] = new_w
                mask[key] = m
                cents[key] = cent_sorted
                labels[key] = lab_sorted

            else:
                clustered[key] = w  # 已经是 Tensor，不需要 detach 除非要断开
                mask[key] = torch.ones_like(w)

        self.mask = mask

        # 优化：重建 _masked_params 缓存列表，配合 _foreach API 使用
        self._param_list_for_masking.clear()
        self._mask_list_for_masking.clear()

        # 只遍历一次 state_dict 来匹配 parameter
        for name, p in self.model.named_parameters():
            if name in self.mask:
                self._param_list_for_masking.append(p)
                self._mask_list_for_masking.append(self.mask[name])

        # 初始化 global_cents 形状
        if self.use_global_cents:
            for k, c in cents.items():
                if (k not in self.global_cents) or (self.global_cents[k].shape != c.shape):
                    self.global_cents[k] = c.detach().clone().to(device)

        return clustered, cents, labels

    # ============================================================
    # 剪枝 mask 应用 (CUDA 优化版)
    # ============================================================
    @torch.no_grad()
    def _apply_prune_mask_inplace(self):
        """
        使用 torch._foreach_mul_ 快速批量应用 mask
        """
        if not self._param_list_for_masking:
            return

        # 这会在 CUDA 上发射一个单一内核处理列表中的所有 Tensor，速度极快
        torch._foreach_mul_(self._param_list_for_masking, self._mask_list_for_masking)

    # ============================================================
    # 构建质心 FedProx 的锚点权重
    # ============================================================
    @torch.no_grad()
    def _build_centroid_prox_target(self) -> Dict[str, torch.Tensor]:
        if (
                self.centroid_reg_lambda <= 0.0
                or not self.use_global_cents
                or not self.global_cents
                or self.cluster_model is None
        ):
            return {}

        _, _, labels = self.cluster_model
        prox_target: Dict[str, torch.Tensor] = {}
        state = self.model.state_dict()
        device = self.device

        for key, idx in labels.items():
            if key not in self.global_cents or key not in state:
                continue

            g = self.global_cents[key].view(-1, 1)  # Ensure [K, 1]
            if g.shape[0] != self.n_clusters:
                continue

            # 使用 index_select 代替简单的 idx 索引，通常在 CUDA 上更高效且显式
            # idx shape [num_params], g shape [K, 1]
            idx_flat = idx.view(-1).long()
            anchor_w = torch.index_select(g, 0, idx_flat).view(state[key].shape)

            prox_target[key] = anchor_w.detach()  # 确保分离

        return prox_target

    # ============================================================
    # 本地训练（优化版：Vectorized Regularization）
    # ============================================================
    def _local_train(self, prox_target: Optional[Dict[str, torch.Tensor]] = None):
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader not initialized")

        use_prox = (
                prox_target is not None
                and len(prox_target) > 0
                and self.centroid_reg_lambda > 0.0
        )

        self.model.train()
        device = self.device
        lam = self.centroid_reg_lambda

        # 优化：预缓存参数对列表，用于 _foreach 计算
        train_params: List[nn.Parameter] = []
        train_anchors: List[torch.Tensor] = []

        if use_prox:
            for name, p in self.model.named_parameters():
                if name in prox_target:
                    train_params.append(p)
                    train_anchors.append(prox_target[name])

        for _ in range(self.epochs):
            if self.apply_mask_every_epoch:
                self._apply_prune_mask_inplace()

            for x, labels in self.client_train_loader:
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()

                if use_prox and train_params:
                    # 优化：使用 torch._foreach API 计算正则项
                    # 1. 计算 diff = p - anchor
                    diffs = torch._foreach_sub(train_params, train_anchors)
                    # 2. 计算 sq = diff^2
                    sq_diffs = torch._foreach_pow(diffs, 2)
                    # 3. 计算每个 Tensor 的 sum，然后 stack 再 sum，或者直接用 foreach_norm
                    reg_val = torch.stack([t.mean() for t in sq_diffs]).sum()

                    loss += lam * reg_val

                loss.backward()
                self.optimizer.step()

    # ============================================================
    # 模型重构
    # ============================================================
    @torch.no_grad()
    def _reconstruct_from_compressed(
            self,
            cents: Dict[str, torch.Tensor],
            labels: Dict[str, torch.Tensor],
            uncompressed: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        reconstructed = {k: v.to(self.device) for k, v in uncompressed.items()}
        device = self.device

        # 获取本地 state 用来参考 shape
        local_state_keys = self.model.state_dict().keys()

        for key, idx in labels.items():
            if key not in cents or key not in local_state_keys:
                continue

            c = cents[key].to(device)
            idx = idx.to(device).long().view(-1)

            # 使用 index_select 重构
            # 这里需要知道原始 shape，暂时只能通过 labels 的长度推断或者存 shape
            # 代码逻辑里 idx 的长度应该等于 numel， reshape 需要原始形状
            # 这里的隐患是如果不传 orig_shape，只能通过 self.model 对应层来推断
            orig_shape = self.model.state_dict()[key].shape

            w_recon = torch.index_select(c, 0, idx).view(orig_shape)
            reconstructed[key] = w_recon

        return reconstructed

    # ============================================================
    # 聚合（深度优化版：Batch Processing）
    # ============================================================
    @torch.no_grad()
    def aggregate(self):
        if not self.neighbor_model_weights_buffer:
            return

        device = self.device

        # 提取数据
        neighbor_states = []
        neighbor_masses = []
        neighbor_cents = []

        for tpl in self.neighbor_model_weights_buffer:
            neighbor_states.append(tpl[0])
            meta = tpl[-1] if isinstance(tpl[-1], dict) else {}
            m = float(meta.get("ps_mass_share", 0.0))
            neighbor_masses.append(m)

            if len(tpl) >= 2 and isinstance(tpl[1], dict):
                neighbor_cents.append(tpl[1])
            else:
                neighbor_cents.append({})

        if not neighbor_states:
            self.neighbor_model_weights_buffer.clear()
            return

        # -------------------------------------------------
        # 1. 首次聚合处理 (Initialization)
        # -------------------------------------------------
        if self.cluster_model is None:
            # 简单的平均初始化
            # 优化：使用 stack mean
            keys = list(neighbor_states[0].keys())  # 转换为 list 确定顺序
            avg_state = {}
            for k in keys:
                # [N, *shape] -> mean(0)
                tensors = [st[k].to(device) for st in neighbor_states]
                avg_state[k] = torch.stack(tensors).mean(dim=0)

            self.model.load_state_dict(avg_state)
            self.cluster_model = self._cluster_and_prune_model_weights()
            self.ps_mass += sum(neighbor_masses)
            self.neighbor_model_weights_buffer.clear()
            return

        # -------------------------------------------------
        # 2. 常规聚合 (Push-Sum Weighted Average)
        # -------------------------------------------------
        local_mass = float(self.ps_mass)
        # 转换为 Tensor 方便计算
        masses_tensor = torch.tensor(neighbor_masses, device=device, dtype=torch.float32)
        total_mass = local_mass + masses_tensor.sum().item()

        if total_mass <= 1e-9:
            total_mass = 1.0  # 避免除零

        self.ps_mass = total_mass  # 更新本地 mass

        keys = list(neighbor_states[0].keys())
        avg_state = {}

        # 批量聚合核心逻辑
        for k in keys:
            # 收集所有邻居的该层权重 [N_neighbors, *shape]
            # 过滤掉 mass <= 0 的情况已经没必要了，乘 0 即可，保持 tensor 形状整齐更利于 CUDA

            # 这里假设所有 neighbor 都有 key k
            nb_tensors = [st[k].to(device) for st in neighbor_states]
            if not nb_tensors: continue

            nb_stack = torch.stack(nb_tensors)  # [N, *shape]

            # 构造 weight shape 用于广播: [N, 1, 1, ...]
            view_shape = [len(neighbor_masses)] + [1] * (nb_stack.ndim - 1)
            w_tensor = masses_tensor.view(*view_shape)

            # 加权和: sum(weights * tensors)
            weighted_sum = (nb_stack * w_tensor).sum(dim=0)

            # 加上本地
            if local_mass > 0:
                weighted_sum += self.model.state_dict()[k] * local_mass

            avg_state[k] = weighted_sum / total_mass

        # -------------------------------------------------
        # 3. Global Cents 更新 (Vectorized)
        # -------------------------------------------------
        if self.use_global_cents and self.cluster_model:
            _, local_cents, _ = self.cluster_model
            new_global_cents = {}

            # 只需要遍历本地有的 key
            for key, c_local in local_cents.items():
                c_local = c_local.to(device)  # [K, 1]

                # 分子分母初始化
                num = c_local * local_mass
                den = local_mass

                # 收集邻居中该层质心
                # 注意：不是所有邻居都有该层质心，或者形状可能不匹配（极少数情况）
                # 为了向量化，我们需要筛选出存在的邻居
                valid_indices = []
                valid_cents = []

                for i, nc in enumerate(neighbor_cents):
                    if key in nc:
                        valid_indices.append(i)
                        valid_cents.append(nc[key].to(device))

                if valid_cents:
                    # stack: [M, K, 1]
                    c_stack = torch.stack(valid_cents)
                    # weights: [M]
                    w_subset = masses_tensor[torch.tensor(valid_indices, device=device)]
                    # reshape weights for broadcast: [M, 1, 1]
                    w_subset_view = w_subset.view(-1, 1, 1)

                    # 向量化累加
                    num += (c_stack * w_subset_view).sum(dim=0)
                    den += w_subset.sum()

                new_global_cents[key] = num / (den + self.sim_eps)

            self.global_cents = new_global_cents

        self.model.load_state_dict(avg_state)
        self.neighbor_model_weights_buffer.clear()

    # ============================================================
    # 其他辅助方法保持不变 (Prepare meta, set init, send, receive, train)
    # ============================================================
    def _prepare_maturity_meta(self) -> Dict[str, Any]:
        d_out = self.k_push + 1
        share = float(self.ps_mass) / float(d_out)
        self.ps_mass = share
        return {
            "version_meta": "cadfedfilter_meta_v1",
            "sender_id": self.id,
            "version": self.local_version,
            "sender_time": self.last_update_time,
            "ps_mass_share": share,
        }

    def train(self):
        # 保持原有逻辑流
        self.cluster_model = self._cluster_and_prune_model_weights()
        prox_target = self._build_centroid_prox_target()
        self._local_train(prox_target=prox_target)
        self.cluster_model = self._cluster_and_prune_model_weights()

    def set_init_model(self, model: nn.Module):
        self.model = deepcopy(model).to(self.device)
        self.cluster_model = None
        self.ps_mass = 1.0
        self.global_cents = {}
        self.mask.clear()
        self._param_list_for_masking = []
        self._mask_list_for_masking = []

    def send_model(self):
        if self.cluster_model is None:
            self.cluster_model = self._cluster_and_prune_model_weights()

        clustered_state_dict, cents, labels = self.cluster_model
        # 优化：字典推导式很快，无需改动
        uncompressed = {k: v for k, v in clustered_state_dict.items() if k not in cents}

        payload = {
            "cents": cents,
            "labels": labels,
            "uncompressed": uncompressed,
        }
        meta = self._prepare_maturity_meta()
        return payload, meta

    @torch.no_grad()
    def receive_neighbor_model(self, neighbor_payload):
        # 保持原样，解码逻辑主要在于 _reconstruct_from_compressed
        if isinstance(neighbor_payload, tuple) and len(neighbor_payload) >= 2:
            compressed = neighbor_payload[0]
            meta = neighbor_payload[-1]
            if isinstance(compressed, dict) and "cents" in compressed:
                reconstructed = self._reconstruct_from_compressed(
                    compressed["cents"], compressed["labels"], compressed["uncompressed"]
                )
                new_payload = (reconstructed, compressed["cents"], compressed["labels"], meta)
            else:
                new_payload = neighbor_payload
        else:
            new_payload = neighbor_payload
        super().receive_neighbor_model(new_payload)