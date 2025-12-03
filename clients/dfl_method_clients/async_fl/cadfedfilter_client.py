from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from clients.client import Client
from utils.kmeans import TorchKMeans


class CADFedFilterClient(Client):
    """
    Centroid-Guided Client in Asynchronous Decentralized FL
    版本：A — 质心空间 FedProx 正则，用于缓解 non-IID

    核心思路：
      - 用 KMeans 把模型权重压缩为 {centroids, labels}（通信压缩 + 剪枝）
      - 每个客户端维护一个 global_cents，作为“全局 codebook 质心”的估计
      - 本地训练时，在质心空间做 FedProx 类正则：
          当前 weights 通过本地 labels 对齐到 global_cents 解码得到锚点权重 W_anchor
          L_total = CE_loss + λ * ||W - W_anchor||^2
      - Push-sum 仍然负责解决系统异质性（网络不对称 / 异步）
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        hp = dict(hyperparam)
        super().__init__(client_id, dataset_index, full_dataset, hp, device)

        # ========== 质心压缩 / 剪枝相关 ==========
        # cluster_model = (clustered_state_dict, cents, labels)
        self.cluster_model: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None
        self.mask: Dict[str, torch.Tensor] = {}

        self.n_clusters: int = int(hp.get("n_clusters", 16))
        self.epochs: int = int(hp.get("epochs", 1))

        # 数值稳定 eps
        self.sim_eps: float = float(hp.get("sim_eps", 1e-8))
        # 每个 epoch 是否重新应用剪枝 mask
        self.apply_mask_every_epoch: bool = bool(hp.get("apply_mask_every_epoch", True))

        # ========== Push-sum 质量 ==========
        self.ps_mass: float = 1.0

        # ========== 全局质心（质心空间的“全局锚点”） ==========
        # 是否使用 global_cents 来做：
        #   - KMeans 初始化
        #   - FedProx 正则锚点
        self.use_global_cents: bool = bool(hp.get("use_global_cents", True))
        self.global_cents: Dict[str, torch.Tensor] = {}

        # 质心空间 FedProx 正则强度 λ；=0 表示关闭该功能
        self.centroid_reg_lambda: float = float(hp.get("centroid_reg_lambda", 1e-1))

    # ============================================================
    # KMeans 聚类辅助
    # ============================================================
    @staticmethod
    def _sort_centroids_and_remap(c1d: torch.Tensor, lbl: torch.Tensor):
        """
        把质心按数值排序，并重映射 labels，保证质心顺序在不同客户端更可比。
        """
        c = c1d.view(-1).detach()
        order = torch.argsort(c)
        sorted_c = c[order].view(-1, 1)

        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        new_labels = old2new[lbl]

        return sorted_c, new_labels

    def _cluster_and_prune_model_weights(self):
        """
        对当前 self.model 做 KMeans 质心压缩 + 掩码剪枝。
        返回:
          - clustered: 压缩后（用质心重构）的权重字典
          - cents:     {layer_key: [K,1]} 每层的质心
          - labels:    {layer_key: [num_params]} 每个参数对应的质心索引
        额外副作用：
          - 更新 self.mask
          - 若 use_global_cents=True 且 global_cents 尚未初始化，对其做形状初始化
        """
        clustered: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}
        cents: Dict[str, torch.Tensor] = {}
        labels: Dict[str, torch.Tensor] = {}

        state = self.model.state_dict()
        device = self.device

        for key, w in state.items():
            # 简单示意：只对非 BN / 非 downsample 的 weight 做聚类
            if "weight" in key and "bn" not in key and "downsample" not in key:
                orig = w.shape
                flat = w.detach().view(-1, 1).to(device)

                kmeans = TorchKMeans(
                    n_clusters=self.n_clusters,
                    is_sparse=True,
                    init_centroids=None,
                )

                # 如果有全局质心，对应层作为 KMeans 初始质心（shape 匹配才使用）
                if self.use_global_cents:
                    g = self.global_cents.get(key, None)
                    if g is not None:
                        g_init = g.view(-1, 1) if g.dim() == 1 else g
                        if g_init.shape[0] == self.n_clusters:
                            kmeans.init_centroids = g_init.to(flat.device)

                kmeans.fit(flat)

                cent_sorted, lab_sorted = self._sort_centroids_and_remap(
                    kmeans.centroids.view(-1, 1), kmeans.labels_
                )
                cent_sorted = cent_sorted.to(device)

                new_w = cent_sorted[lab_sorted].view(orig)

                # 剪枝：把“质心为 0”的 cluster 对应的权重 mask 掉
                is_zero = cent_sorted.view(-1) == 0
                m = (is_zero[lab_sorted].view(orig) == 0)

                clustered[key] = new_w
                mask[key] = m.bool()
                cents[key] = cent_sorted
                labels[key] = lab_sorted.view(-1).to(device)

            else:
                # 非聚类层：原样保留，不剪枝
                clustered[key] = w.detach().to(device)
                mask[key] = torch.ones_like(w, dtype=torch.bool, device=device)

        self.mask = mask

        # 若尚未有 global_cents，用本地的形状初始化（只做 shape 对齐，不强行同步值）
        if self.use_global_cents:
            for k, c in cents.items():
                if (k not in self.global_cents) or (self.global_cents[k].shape != c.shape):
                    self.global_cents[k] = c.detach().clone().to(device)

        return clustered, cents, labels

    # ============================================================
    # 剪枝 mask 应用
    # ============================================================
    def _apply_prune_mask_inplace(self):
        """
        在当前模型参数上原地应用剪枝 mask。
        """
        if not self.mask:
            return

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.mask:
                    p.mul_(self.mask[name].to(self.device))

    # ============================================================
    # 构建质心 FedProx 的锚点权重
    # ============================================================
    def _build_centroid_prox_target(self) -> Dict[str, torch.Tensor]:
        """
        使用 global_cents + 本地 labels 构造一个“锚点权重”：
          anchor[key] = decode(global_cents[key], labels[key])

        本地训练时，我们会对 (W[key] - anchor[key])^2 加一个 λ 正则。
        若当前没有有效的 global_cents 或 lambda=0，则返回空字典。
        """
        if (
            self.centroid_reg_lambda <= 0.0
            or not self.use_global_cents
            or len(self.global_cents) == 0
            or self.cluster_model is None
        ):
            return {}

        _, local_cents, labels = self.cluster_model
        prox_target: Dict[str, torch.Tensor] = {}
        state = self.model.state_dict()
        device = self.device

        for key, idx in labels.items():
            if key not in self.global_cents:
                continue

            g = self.global_cents[key]  # [K,1]
            if g.dim() == 1:
                g = g.view(-1, 1)

            # 要求质心数匹配
            if g.shape[0] != self.n_clusters:
                continue

            idx_flat = idx.to(device=device, dtype=torch.long).view(-1)
            g_expanded = g.to(device)[idx_flat]  # [num_params, 1]

            if key not in state:
                continue

            orig_shape = state[key].shape
            anchor_w = g_expanded.view(orig_shape).detach()
            prox_target[key] = anchor_w

        return prox_target

    # ============================================================
    # 本地训练（带质心 FedProx 正则）
    # ============================================================
    def _local_train(self, prox_target: Optional[Dict[str, torch.Tensor]] = None):
        """
        本地 SGD 训练，在原有损失上加一个质心空间 FedProx 正则：
          L = L_task + λ * Σ ||W[key] - anchor[key]||^2

        其中 anchor 由 global_cents + labels 解码得到。
        """
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

        for _ in range(self.epochs):
            if self.apply_mask_every_epoch and len(self.mask) > 0:
                self._apply_prune_mask_inplace()

            for x, labels in self.client_train_loader:
                x = x.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()

                # 质心 FedProx 正则
                if use_prox:
                    reg = 0.0
                    for name, p in self.model.named_parameters():
                        if name in prox_target:
                            anchor = prox_target[name].to(device)
                            reg = reg + (p - anchor).pow(2).mean()
                    loss = loss + lam * reg

                loss.backward()
                self.optimizer.step()

    # ============================================================
    # 模型重构（解码压缩）
    # ============================================================
    def _reconstruct_from_compressed(
        self,
        cents: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        uncompressed: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        从压缩表示（centroids + labels + uncompressed）恢复完整 state_dict。
        """
        reconstructed: Dict[str, torch.Tensor] = {}

        # 非压缩层先写入
        for k, v in uncompressed.items():
            reconstructed[k] = v.to(self.device)

        local_state = self.model.state_dict()

        # 压缩层：质心查表 + reshape
        for key, idx in labels.items():
            if key not in cents:
                continue

            c = cents[key].to(self.device)      # [K,1]
            idx = idx.to(self.device).long()    # [num_params]

            if key not in local_state:
                continue

            orig_shape = local_state[key].shape
            w_recon = c[idx].view(orig_shape)
            reconstructed[key] = w_recon

        return reconstructed

    # ============================================================
    # Push-sum meta 打包
    # ============================================================
    def _prepare_maturity_meta(self) -> Dict[str, Any]:
        """
        标准 push-sum 质量拆分：
          - 把当前 ps_mass 均分成 (k_push + 1) 份，一份留给自己，其余发给邻居
        """
        d_out = self.k_push + 1
        share = float(self.ps_mass) / float(d_out)
        # 自己也只留一份 share
        self.ps_mass = share

        return {
            "version_meta": "cadfedfilter_meta_v1",
            "sender_id": self.id,
            "version": self.local_version,
            "sender_time": self.last_update_time,
            "ps_mass_share": share,
        }

    # ============================================================
    # 聚合：标准 push-sum 共识 + 质心空间 global_cents 更新
    # ============================================================
    @torch.no_grad()
    def aggregate(self):
        """
        去中心化异步场景下的模型聚合：
          - 仍然是标准 push-sum 加权平均（无偏）
          - 使用压缩模型重构后的 state_dict 做加权
          - 使用本地 & 邻居的质心（cents）更新 global_cents（质心空间共识）
        """
        if len(self.neighbor_model_weights_buffer) == 0:
            return

        # ---------- 首次聚合：还没有 cluster_model ----------
        if self.cluster_model is None:
            neighbor_states = [tpl[0] for tpl in self.neighbor_model_weights_buffer]
            if len(neighbor_states) == 0:
                self.neighbor_model_weights_buffer.clear()
                return

            keys = set(neighbor_states[0].keys())
            avg_state: Dict[str, torch.Tensor] = {}
            for k in keys:
                avg_state[k] = torch.stack(
                    [st[k].to(self.device) for st in neighbor_states]
                ).mean(dim=0)

            # 用邻居平均初始化自己的模型
            self.model.load_state_dict(avg_state)

            # 初始化质心模型（也会顺带初始化 global_cents 形状）
            self.cluster_model = self._cluster_and_prune_model_weights()

            # 标准 push-sum：累加收到的 mass
            recv_mass = sum(
                float(tpl[-1].get("ps_mass_share", 0.0))
                for tpl in self.neighbor_model_weights_buffer
                if isinstance(tpl[-1], dict)
            )
            self.ps_mass += float(recv_mass)

            self.neighbor_model_weights_buffer.clear()
            return

        # ---------- 常规聚合（无偏 push-sum 共识） ----------
        neighbor_states = []
        neighbor_masses_raw = []
        neighbor_cents_list = []

        for tpl in self.neighbor_model_weights_buffer:
            st = tpl[0]
            neighbor_states.append(st)

            meta = tpl[-1] if isinstance(tpl[-1], dict) else {}
            neighbor_masses_raw.append(float(meta.get("ps_mass_share", 0.0)))

            # 若邻居带了质心信息，保留以便更新 global_cents
            if len(tpl) >= 2 and isinstance(tpl[1], dict):
                neighbor_cents_list.append(tpl[1])
            else:
                neighbor_cents_list.append({})

        if len(neighbor_states) == 0:
            self.neighbor_model_weights_buffer.clear()
            return

        keys = set(neighbor_states[0].keys())
        local_mass = float(self.ps_mass)
        local_state = self.model.state_dict()

        # 不做任何 reweight，保持无偏 push-sum
        eff_masses = neighbor_masses_raw

        # 若所有邻居质量为 0 且本地质量也为 0，直接不聚合
        if sum(eff_masses) == 0.0 and local_mass <= 0.0:
            self.neighbor_model_weights_buffer.clear()
            return

        # ---------- push-sum mass 更新 ----------
        total_mass = local_mass + sum(eff_masses)
        if total_mass <= 0:
            total_mass = 1.0
        self.ps_mass = float(total_mass)

        # ---------- 计算混合模型 avg_state ----------
        avg_state: Dict[str, torch.Tensor] = {}
        device = self.device

        for k in keys:
            acc = None

            # 邻居贡献
            for st, m in zip(neighbor_states, eff_masses):
                if m <= 0.0 or k not in st:
                    continue
                t = st[k].to(device)
                acc = t.mul(m) if acc is None else acc.add(t, alpha=m)

            # 本地贡献
            if local_mass > 0.0:
                t_local = local_state[k].to(device)
                acc = t_local.mul(local_mass) if acc is None else acc.add(t_local, alpha=local_mass)

            if acc is None:
                avg_state[k] = local_state[k].to(device)
            else:
                avg_state[k] = acc.div(total_mass)

        # ---------- 更新 global_cents（用原始 mass） ----------
        if self.use_global_cents and self.cluster_model is not None:
            _, local_cents, _ = self.cluster_model
            new_global_cents: Dict[str, torch.Tensor] = {}

            for key, c_local in local_cents.items():
                c_local = c_local.to(device)

                num = c_local * local_mass
                den = local_mass

                for c_nb, m_eff in zip(neighbor_cents_list, eff_masses):
                    if m_eff <= 0.0:
                        continue
                    if key not in c_nb:
                        continue
                    num = num + m_eff * c_nb[key].to(device)
                    den = den + m_eff

                new_global_cents[key] = num / (den + self.sim_eps)

            self.global_cents = new_global_cents

        # ---------- 把模型更新为 avg_state ----------
        self.model.load_state_dict(avg_state)

        # 下一轮 train 会重新聚类，这里不强制刷新 cluster_model
        self.neighbor_model_weights_buffer.clear()

    # ============================================================
    # 训练接口：在 train() 里处理质心初始化 + FedProx 正则
    # ============================================================
    def train(self):
        """
        一轮本地训练流程：
          1) 根据当前模型做聚类，得到 cluster_model = (clustered, cents, labels)
          2) 基于 global_cents + labels 构造 FedProx 锚点权重 prox_target
          3) 在锚点约束下做本地 SGD 训练
          4) 用新模型重新聚类，更新 cluster_model（用于后续压缩 + global_cents 更新）
        """
        # 1: 根据当前模型聚类（也会设置剪枝 mask & 初始化 global_cents 形状）
        self.cluster_model = self._cluster_and_prune_model_weights()

        # 2: 基于 global_cents 构造 FedProx 锚点
        prox_target = self._build_centroid_prox_target()

        # 3: 本地训练（带或不带质心正则）
        self._local_train(prox_target=prox_target)

        # 4: 用新模型重新聚类，更新 cluster_model
        self.cluster_model = self._cluster_and_prune_model_weights()

    # ============================================================
    # 模型发送 & 接收（压缩 + meta）
    # ============================================================
    def set_init_model(self, model: nn.Module):
        """
        初始化该客户端的模型副本 & 质心相关状态。
        """
        self.model = deepcopy(model).to(self.device)
        self.cluster_model = None
        self.ps_mass = 1.0
        self.global_cents = {}
        self.mask.clear()

    def send_model(self):
        """
        发送压缩模型：
          payload = {cents, labels, uncompressed}
          meta    = {ps_mass_share, ...}
        """
        # 确保有最新的 cluster_model
        if self.cluster_model is None:
            clustered_state_dict, cents, labels = self._cluster_and_prune_model_weights()
            self.cluster_model = (clustered_state_dict, cents, labels)
        else:
            clustered_state_dict, cents, labels = self.cluster_model

        # 未压缩层（不在 cents 里的层）
        uncompressed = {k: v for k, v in clustered_state_dict.items() if k not in cents}

        payload = {
            "cents": cents,
            "labels": labels,
            "uncompressed": uncompressed,
        }

        meta = self._prepare_maturity_meta()
        return payload, meta

    def receive_neighbor_model(self, neighbor_payload):
        """
        解码压缩模型：
          原始： (state_dict, meta)
          压缩： ({"cents","labels","uncompressed"}, meta)
          解码后统一为： (state_dict, cents, labels, meta)
        """
        if isinstance(neighbor_payload, tuple) and len(neighbor_payload) >= 2:
            compressed = neighbor_payload[0]
            meta = neighbor_payload[-1]

            if isinstance(compressed, dict) and "cents" in compressed:
                nb_cents = compressed["cents"]
                nb_labels = compressed["labels"]
                nb_uncompressed = compressed["uncompressed"]

                reconstructed = self._reconstruct_from_compressed(
                    nb_cents, nb_labels, nb_uncompressed
                )
                new_payload = (reconstructed, nb_cents, nb_labels, meta)
            else:
                new_payload = neighbor_payload
        else:
            new_payload = neighbor_payload

        super().receive_neighbor_model(new_payload)
