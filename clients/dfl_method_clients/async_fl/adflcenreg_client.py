from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from clients.client import Client
from utils.kmeans import TorchKMeans

# ADFL-CenReg
class ADFLCenRegClient(Client):
    """
    Asynchronous Decentralize Federated Learning with Centroid Regularization
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        hp = dict(hyperparam)
        super().__init__(client_id, dataset_index, full_dataset, hp, device)

        self.cluster_model: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None

        self.mask: Dict[str, torch.Tensor] = {}
        self._param_list_for_masking: List[nn.Parameter] = []
        self._mask_list_for_masking: List[torch.Tensor] = []

        self.n_clusters: int = int(hp.get("n_clusters", 16))
        self.epochs: int = int(hp.get("epochs", 1))
        self.sim_eps: float = float(hp.get("sim_eps", 1e-8))
        self.apply_mask_every_epoch: bool = bool(hp.get("apply_mask_every_epoch", True))
        self.ps_mass: float = 1.0
        self.use_global_cents: bool = bool(hp.get("use_global_cents", True))
        self.global_cents: Dict[str, torch.Tensor] = {}
        self.lambda_reg: float = float(hp.get("lambda_reg", 1e-1))

    @staticmethod
    def _sort_centroids_and_remap(c1d: torch.Tensor, lbl: torch.Tensor):
        c = c1d.view(-1)
        order = torch.argsort(c)
        sorted_c = c[order].view(-1, 1)

        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        new_labels = old2new[lbl]

        return sorted_c, new_labels

    @torch.no_grad()
    def _cluster_and_prune_model_weights(self):
        mask: Dict[str, torch.Tensor] = {}
        cents: Dict[str, torch.Tensor] = {}
        labels: Dict[str, torch.Tensor] = {}

        state = self.model.state_dict()
        device = self.device

        for key, w in state.items():
            # 筛选条件保持不变
            if "weight" in key and "bn" not in key and "downsample" not in key:
                orig_shape = w.shape
                flat = w.view(-1, 1)
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

                new_w = torch.index_select(cent_sorted, 0, lab_sorted.view(-1)).view(orig_shape)

                zero_indices = torch.nonzero(cent_sorted.view(-1) == 0, as_tuple=True)[0]

                if zero_indices.numel() > 0:
                    m = (new_w != 0).to(dtype=w.dtype)
                else:
                    m = torch.ones_like(w)

                mask[key] = m
                cents[key] = cent_sorted
                labels[key] = lab_sorted

            else:
                mask[key] = torch.ones_like(w)

        self.mask = mask

        self._param_list_for_masking.clear()
        self._mask_list_for_masking.clear()

        for name, p in self.model.named_parameters():
            if name in self.mask:
                self._param_list_for_masking.append(p)
                self._mask_list_for_masking.append(self.mask[name])

        if self.use_global_cents:
            for k, c in cents.items():
                if (k not in self.global_cents) or (self.global_cents[k].shape != c.shape):
                    self.global_cents[k] = c.detach().clone().to(device)

        return cents, labels

    @torch.no_grad()
    def _apply_prune_mask_inplace(self):
        if not self._param_list_for_masking:
            return

        torch._foreach_mul_(self._param_list_for_masking, self._mask_list_for_masking)

    @torch.no_grad()
    def _build_centroid_prox_target(self) -> Dict[str, torch.Tensor]:
        if (
                self.lambda_reg <= 0.0
                or not self.use_global_cents
                or not self.global_cents
                or self.cluster_model is None
        ):
            return {}

        _, labels = self.cluster_model
        prox_target: Dict[str, torch.Tensor] = {}
        state = self.model.state_dict()

        for key, idx in labels.items():
            if key not in self.global_cents or key not in state:
                continue

            g = self.global_cents[key].view(-1, 1)
            if g.shape[0] != self.n_clusters:
                continue

            idx_flat = idx.view(-1).long()
            anchor_w = torch.index_select(g, 0, idx_flat).view(state[key].shape)

            prox_target[key] = anchor_w.detach()

        return prox_target

    def _local_train(self, prox_target: Optional[Dict[str, torch.Tensor]] = None):
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader not initialized")

        use_prox = (
                prox_target is not None
                and len(prox_target) > 0
                and self.lambda_reg > 0.0
        )

        self.model.train()
        device = self.device
        lam = self.lambda_reg

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
                    diffs = torch._foreach_sub(train_params, train_anchors)
                    sq_diffs = torch._foreach_pow(diffs, 2)
                    reg_val = torch.stack([t.mean() for t in sq_diffs]).sum()

                    loss += lam * reg_val

                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def _reconstruct_from_compressed(
            self,
            cents: Dict[str, torch.Tensor],
            labels: Dict[str, torch.Tensor],
            uncompressed: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        reconstructed = {k: v.to(self.device) for k, v in uncompressed.items()}
        device = self.device

        local_state_keys = self.model.state_dict().keys()

        for key, idx in labels.items():
            if key not in cents or key not in local_state_keys:
                continue

            c = cents[key].to(device)
            idx = idx.to(device).long().view(-1)

            orig_shape = self.model.state_dict()[key].shape

            w_recon = torch.index_select(c, 0, idx).view(orig_shape)
            reconstructed[key] = w_recon

        return reconstructed

    @torch.no_grad()
    def mask_stats(self) -> Dict[str, float]:
        """
        返回全局 & 按层的 mask 统计：
        - global_density: 全局保留率(1占比)
        - global_sparsity: 全局稀疏率(0占比)
        - per_layer_density: 每层保留率
        - per_layer_sparsity: 每层稀疏率
        """
        if not self.mask:
            return {
                "global_density": 1.0,
                "global_sparsity": 0.0,
            }

        total_ones = 0.0
        total_elems = 0.0

        per_layer_density = {}
        per_layer_sparsity = {}

        for name, m in self.mask.items():
            # m 是 float/bfloat/half 的 0/1 张量（你代码里是 w.dtype）
            ones = float(m.sum().item())
            elems = float(m.numel())
            dens = ones / max(elems, 1.0)
            spar = 1.0 - dens

            per_layer_density[name] = dens
            per_layer_sparsity[name] = spar

            total_ones += ones
            total_elems += elems

        global_density = total_ones / max(total_elems, 1.0)
        global_sparsity = 1.0 - global_density

        return {
            "global_density": global_density,
            "global_sparsity": global_sparsity,
            "per_layer_density": per_layer_density,
            "per_layer_sparsity": per_layer_sparsity,
        }

    @torch.no_grad()
    def aggregate(self):
        """
        Push-sum 聚合 + 质心 global_cents 更新
        - BN 相关参数（key 中含 "bn"）完全本地化，不参与聚合（FedBN 风格）
        - 仅对浮点/复数参数做加权平均
        - 非浮点（long/bool）保持本地，避免 dtype 问题
        """
        if not self.neighbor_model_weights_buffer:
            return

        device = self.device

        # --------- 收集邻居的 state / mass / cents ---------
        neighbor_states: List[Dict[str, torch.Tensor]] = []
        neighbor_masses: List[float] = []
        neighbor_cents: List[Dict[str, torch.Tensor]] = []

        for tpl in self.neighbor_model_weights_buffer:
            st = tpl[0]
            neighbor_states.append(st)

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

        local_state = self.model.state_dict()

        # =====================================================
        # 1) 首次聚合：还没有 cluster_model，用邻居初始化模型
        #    BN 参数全部保留本地，不参与初始化平均
        # =====================================================
        if self.cluster_model is None:
            avg_state: Dict[str, torch.Tensor] = {}

            for k, v_local in local_state.items():
                t_local = v_local.to(device)

                # --- BN 参数全部本地化 ---
                if "bn" in k:
                    avg_state[k] = t_local
                    continue

                # 非 BN：用邻居做简单平均（不包含本地）
                tensors = [st[k].to(device) for st in neighbor_states if k in st]
                if not tensors:
                    avg_state[k] = t_local
                    continue

                t0 = tensors[0]
                if t0.is_floating_point() or t0.is_complex():
                    avg_state[k] = torch.stack(tensors, dim=0).mean(dim=0)
                else:
                    # 非浮点（long/bool 等）保持本地
                    avg_state[k] = t_local

            # 更新模型
            self.model.load_state_dict(avg_state)
            # 初始化质心模型（会同时初始化 mask / global_cents 形状）
            self.cluster_model = self._cluster_and_prune_model_weights()
            # push-sum 质量累加邻居 mass
            self.ps_mass += sum(neighbor_masses)

            self.neighbor_model_weights_buffer.clear()
            return

        # =====================================================
        # 2) 常规聚合：Push-sum 加权平均（BN 本地化）
        # =====================================================
        local_mass = float(self.ps_mass)
        masses_tensor = torch.tensor(neighbor_masses, device=device, dtype=torch.float32)
        total_mass = local_mass + masses_tensor.sum().item()
        if total_mass <= 1e-9:
            total_mass = 1.0
        self.ps_mass = total_mass

        avg_state: Dict[str, torch.Tensor] = {}

        # 按本地 state 的 key 遍历，保证 BN 也被写回
        for k, v_local in local_state.items():
            t_local = v_local.to(device)

            # ---- BN 全部本地，不参与聚合（FedBN 风格）----
            if "bn" in k:
                avg_state[k] = t_local
                continue

            # 收集有该参数的邻居及其索引
            nb_tensors: List[torch.Tensor] = []
            valid_indices: List[int] = []
            for i, st in enumerate(neighbor_states):
                if k in st:
                    nb_tensors.append(st[k].to(device))
                    valid_indices.append(i)

            if not nb_tensors:
                # 没有邻居提供该参数，直接保留本地
                avg_state[k] = t_local
                continue

            # 只对浮点/复数做 push-sum 平均
            if t_local.is_floating_point() or t_local.is_complex():
                nb_stack = torch.stack(nb_tensors, dim=0)  # [M, *shape]
                idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
                w_subset = masses_tensor.index_select(0, idx_tensor)  # [M]
                view_shape = [len(nb_tensors)] + [1] * (nb_stack.ndim - 1)
                w_view = w_subset.view(*view_shape)  # [M,1,1,...]

                weighted_sum = (nb_stack * w_view).sum(dim=0)
                if local_mass > 0.0:
                    weighted_sum = weighted_sum + t_local * local_mass

                avg_state[k] = weighted_sum / total_mass
            else:
                # 非浮点：保持本地
                avg_state[k] = t_local

        # =====================================================
        # 3) 更新 global_cents（仅卷积/FC 权重参与）
        # =====================================================
        if self.use_global_cents and self.cluster_model:
            local_cents, _ = self.cluster_model
            new_global_cents: Dict[str, torch.Tensor] = {}

            for key, c_local in local_cents.items():
                c_local = c_local.to(device)

                num = c_local * local_mass
                den = local_mass

                valid_indices = []
                valid_cents = []
                for i, nc in enumerate(neighbor_cents):
                    if key in nc:
                        valid_indices.append(i)
                        valid_cents.append(nc[key].to(device))

                if valid_cents:
                    c_stack = torch.stack(valid_cents, dim=0)  # [M,K,1]
                    idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
                    w_subset = masses_tensor.index_select(0, idx_tensor)  # [M]
                    w_view = w_subset.view(-1, 1, 1)  # [M,1,1]

                    num = num + (c_stack * w_view).sum(dim=0)
                    den = den + w_subset.sum()

                new_global_cents[key] = num / (den + self.sim_eps)

            self.global_cents = new_global_cents

        # =====================================================
        # 4) 写回模型 & 清空缓冲
        # =====================================================
        self.model.load_state_dict(avg_state)
        self.neighbor_model_weights_buffer.clear()

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

        stats = self.mask_stats()
        print("density:", stats["global_density"], "sparsity:", stats["global_sparsity"])

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

        cents, labels = self.cluster_model
        state_dict = self.model.state_dict()
        uncompressed = {k: v for k, v in state_dict.items() if k not in cents}

        payload = {
            "cents": cents,
            "labels": labels,
            "uncompressed": uncompressed,
        }
        meta = self._prepare_maturity_meta()
        return payload, meta

    @torch.no_grad()
    def receive_neighbor_model(self, neighbor_payload):
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