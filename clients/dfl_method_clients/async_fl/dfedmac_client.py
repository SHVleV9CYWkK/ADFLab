from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F

from clients.client import Client
from utils.kmeans import TorchKMeans


def _cfd_distance(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """
    简化版 CFD：两个质心向量的平均 L2 距离（已按维度对齐）。
    c1, c2: [K, 1]
    返回标量 Tensor
    """
    return torch.mean((c1.view(-1, 1) - c2.view(-1, 1)) ** 2)


class DFedMACClient(Client):
    """
    异步版 DFedMAC：
      - 接收邻居的 (clustered_state_dict, centroids_dict, labels_dict, meta) 作为“老师”
      - 本地训练时，基于 CFD 相似度 + 老师层级成熟度，构造对齐损失
      - 训练完成后，做一次聚类/剪枝，更新本地成熟度（C-SWAG × 稳定度），并发送 4 元组 + meta
      - 与异步协调器配合：不依赖全局轮次，只用 client.local_version / last_update_time

    重要约定：
      - 该客户端不做“收到即聚合”权重平均，因此覆盖基类行为：强制 fuse_on_receive=False
      - 邻居缓冲保存 3/4 元组（老师表示），仅在下一次本地训练中使用；训练结束后清空
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        # —— 基本开关 / 超参 —— #
        # 这类方法是“对齐型”方法；保持一个显式开关，便于消融
        self.is_align: bool = bool(hyperparam.get("is_align", True))
        self.lambda_alignment: float = float(hyperparam.get("lambda_alignment", 0.1))
        self.n_clusters: int = int(hyperparam.get("n_clusters", 16))

        # 接收端：覆盖基类默认，异步对齐不做即时融合
        self.fuse_on_receive: bool = False
        self.buffer_limit: int = int(hyperparam.get('buffer_limit', 20))

        # —— 成熟度相关超参 —— #
        self.maturity_window: int = int(hyperparam.get('maturity_window', 5))  # C-SWAG 窗口
        self.beta_drift: float = float(hyperparam.get('beta_drift', 1.0))      # 稳定度：漂移项系数
        self.beta_invar: float = float(hyperparam.get('beta_invar', 1.0))      # 稳定度：簇内方差项系数
        self.beta_mask: float = float(hyperparam.get('beta_mask', 0.5))        # 稳定度：mask 翻转率项系数
        self.maturity_eps: float = float(hyperparam.get('maturity_eps', 1e-8))

        # —— 教师柔性降权 —— #
        self.teacher_gamma: float = float(hyperparam.get('teacher_gamma', 1.0))
        self.teacher_blend: float = float(hyperparam.get('teacher_blend', 0.6))  # 0~1

        # 历史缓存：用于本地成熟度估计（C-SWAG 与 mask 翻转）
        self._local_hist: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.maturity_window))  # 每层质心历史
        self._local_mask_hist: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2))                # 每层最近 mask

        # DKM 相关：这里不引入额外类，直接在损失里用 KMeans + 教师混合质心重构
        self.dkm_layers: Dict[str, dict] = {}  # 仅用于记录该层是否可聚类

        # 缓存：训练后的一次聚类结果，供 send_model 复用
        self.cluster_model: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None
        self.mask: Dict[str, torch.Tensor] = {}

        # 老师信息缓存（由 _all_teacher_info() 在每次 train() 前构建）
        self.teacher_info_list: List[dict] = []

        # （可选）统计
        self.local_round: int = 0  # 与 local_version 类似，仅作为可读字段

    # -----------------------------
    # 工具：质心排序 + 标签重映射
    # -----------------------------
    @staticmethod
    def _sort_centroids_and_remap(centroids_1d: torch.Tensor, labels_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = centroids_1d.view(-1).detach()
        order = torch.argsort(c)  # 升序
        sorted_c = c[order].view(-1, 1)
        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        new_labels = old2new[labels_1d]
        return sorted_c, new_labels

    @staticmethod
    def _cluster_intra_var(flat_weights: torch.Tensor, labels: torch.Tensor, K: int, eps: float = 1e-8) -> torch.Tensor:
        vars_out = flat_weights.new_zeros(K, 1)
        for k in range(K):
            idx = (labels == k)
            if idx.any():
                w = flat_weights[idx].view(-1)
                vars_out[k, 0] = torch.var(w, unbiased=False)
            else:
                vars_out[k, 0] = eps
        return vars_out + eps

    # -----------------------------
    # 本地历史更新（用于成熟度）
    # -----------------------------
    def _update_local_history(self, centroids_dict: Dict[str, torch.Tensor], mask_dict: Dict[str, torch.Tensor]) -> None:
        for layer, c in centroids_dict.items():
            self._local_hist[layer].append(c.detach().cpu())
        for layer, m in mask_dict.items():
            self._local_mask_hist[layer].append(m.detach().cpu())

    # -----------------------------
    # C-SWAG 置信度 & 稳定度
    # -----------------------------
    def _c_swag_precision(self, hist_deque: deque) -> torch.Tensor:
        if len(hist_deque) < 2:
            k = hist_deque[-1].shape[0]
            return hist_deque[-1].new_ones(k, 1)
        stack = torch.stack(list(hist_deque), dim=0)  # [T,K,1]
        var = torch.var(stack, dim=0, unbiased=False) # [K,1]
        return 1.0 / (var + self.maturity_eps)

    def _stability_scores(self, layer_key: str, centroids_now: torch.Tensor, labels_now: torch.Tensor, mask_now: torch.Tensor) -> torch.Tensor:
        device = centroids_now.device
        # 漂移：当前与上一帧质心差
        if len(self._local_hist[layer_key]) >= 1:
            prev_c = self._local_hist[layer_key][-1].to(device)
            drift = torch.abs(centroids_now - prev_c)  # [K,1]
        else:
            drift = torch.zeros_like(centroids_now)

        # 簇内方差（当前）
        flat_w = self.model.state_dict()[layer_key].to(device).view(-1, 1).detach()
        K = centroids_now.shape[0]
        invar = self._cluster_intra_var(flat_w, labels_now, K, eps=self.maturity_eps)  # [K,1]

        # mask 翻转率（层级标量）
        if len(self._local_mask_hist[layer_key]) >= 1:
            prev_m = self._local_mask_hist[layer_key][-1].to(device)
            flips = (prev_m ^ mask_now).float().mean()
        else:
            flips = torch.tensor(0.0, device=device)

        stab = torch.exp(- self.beta_drift * drift - self.beta_invar * invar - self.beta_mask * flips)
        return stab.clamp_min(1e-6)

    def _local_maturity(self, layer_key: str, centroids_now: torch.Tensor, labels_now: torch.Tensor, mask_now: torch.Tensor) -> torch.Tensor:
        lam = self._c_swag_precision(self._local_hist[layer_key]).to(centroids_now.device)  # [K,1]
        stab = self._stability_scores(layer_key, centroids_now, labels_now, mask_now)       # [K,1]
        return lam * stab

    # -----------------------------
    # 聚类 + 剪枝 + 返回质心/标签
    # -----------------------------
    def _cluster_and_prune_model_weights(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        clustered_state_dict: Dict[str, torch.Tensor] = {}
        mask_dict: Dict[str, torch.Tensor] = {}
        centroids_dict: Dict[str, torch.Tensor] = {}
        labels_dict: Dict[str, torch.Tensor] = {}

        state = self.model.state_dict()
        for key, weight in state.items():
            # 仅对 “可聚类的卷积/全连接权重” 进行 kmeans（按你的规则排除 BN/downsample）
            if ('weight' in key) and ('bn' not in key) and ('downsample' not in key):
                original_shape = weight.shape
                flat = weight.detach().view(-1, 1)

                kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)
                kmeans.fit(flat)  # centroids: [K,1], labels_: [N]

                centroids_sorted, labels_sorted = self._sort_centroids_and_remap(
                    kmeans.centroids.view(-1, 1), kmeans.labels_
                )

                # 用质心重构参数
                new_weights = centroids_sorted[labels_sorted].view(original_shape)

                # 以 0 质心代表剪枝；构造 mask
                is_zero = (centroids_sorted.view(-1) == 0)
                mask = (is_zero[labels_sorted].view(original_shape) == 0)

                clustered_state_dict[key] = new_weights
                mask_dict[key] = mask.bool()
                centroids_dict[key] = centroids_sorted  # [K,1]
                labels_dict[key] = labels_sorted.view(-1)  # [N]
                self.dkm_layers[key] = {"enabled": True}
            else:
                clustered_state_dict[key] = weight
                mask_dict[key] = torch.ones_like(weight, dtype=torch.bool)

        # 更新成熟度历史
        self._update_local_history(centroids_dict, mask_dict)
        self.mask = mask_dict
        return clustered_state_dict, centroids_dict, labels_dict

    # -----------------------------
    # 生成层级成熟度元信息（发送用）
    # -----------------------------
    def _prepare_maturity_meta(self) -> Dict[str, Any]:
        # 复用上一轮聚类结果（若为空则临时计算一次）
        if self.cluster_model is None:
            _, centroids_dict, labels_dict = self._cluster_and_prune_model_weights()
        else:
            _, centroids_dict, labels_dict = self.cluster_model

        layer_maturity: Dict[str, float] = {}
        for layer_key, centroids_now in centroids_dict.items():
            device = self.device
            centroids_now = centroids_now.to(device)
            labels_now = labels_dict[layer_key].to(device)
            mask_now = self.mask[layer_key].to(device)
            m_vec = self._local_maturity(layer_key, centroids_now, labels_now, mask_now)  # [K,1]
            layer_maturity[layer_key] = float(m_vec.mean().item())

        meta = {
            "version": "dfedmac_meta_v1",
            "client_id": int(self.id),
            "round": int(self.local_round),  # 本地异步轮
            "layer_maturity": layer_maturity,
            "sender_time": float(self.last_update_time),
            "sender_version": int(self.local_version),
        }
        return meta

    # -----------------------------
    # 构建老师信息（CFD 相似度 + 层级成熟度）
    # -----------------------------
    def _all_teacher_info(self) -> None:
        # 本地质心（基准）
        _, local_centroids_dict, _ = self._cluster_and_prune_model_weights()

        cfd_scores: List[torch.Tensor] = []
        teacher_centroids_dicts: List[Dict[str, torch.Tensor]] = []
        teacher_meta_list: List[Optional[Dict[str, Any]]] = []

        # 兼容 3/4 元组
        for item in self.neighbor_model_weights:
            if isinstance(item, (tuple, list)) and (len(item) in (3, 4)):
                if len(item) == 4:
                    _, teacher_centroids, _, meta = item
                    teacher_meta_list.append(meta)
                else:
                    _, teacher_centroids, _ = item
                    teacher_meta_list.append(None)
                teacher_centroids_dicts.append(teacher_centroids)

        if len(teacher_centroids_dicts) == 0:
            self.teacher_info_list = []
            return

        # 逐老师计算与本地的 CFD 距离（跨层平均）
        per_teacher_vals: List[torch.Tensor] = []
        layer_keys = list(local_centroids_dict.keys())
        device = self.device

        for t_centroids in teacher_centroids_dicts:
            per_layer = []
            for layer_key in layer_keys:
                cfd = _cfd_distance(
                    local_centroids_dict[layer_key].detach().float().to(device),
                    t_centroids[layer_key].detach().float().to(device)
                )
                per_layer.append(cfd)
            per_teacher_vals.append(torch.stack(per_layer).mean())  # 标量

        cfd_tensor = torch.stack(per_teacher_vals, dim=0).to(device)  # [T]

        # 基础相似度权重（CFD 越小越相似 → 权重越大）
        # 先归一化到 [0,1] 再做 softmax(-β·normed)
        min_val, max_val = cfd_tensor.min(), cfd_tensor.max()
        normed = (cfd_tensor - min_val) / (max_val - min_val + 1e-8)
        beta = 2.0
        alpha_base = torch.softmax(-beta * normed, dim=0)  # [T]

        # 读取老师的层级成熟度，并在“同层”内做 min-max 归一
        T = len(teacher_centroids_dicts)
        maturity_mat = torch.ones(len(layer_keys), T, dtype=torch.float, device=device)
        for t_idx, meta in enumerate(teacher_meta_list):
            if isinstance(meta, dict) and ('layer_maturity' in meta):
                lm = meta['layer_maturity']
                for li, layer_key in enumerate(layer_keys):
                    if layer_key in lm:
                        maturity_mat[li, t_idx] = max(float(lm[layer_key]), self.maturity_eps)

        vmin = maturity_mat.min(dim=1, keepdim=True).values
        vmax = maturity_mat.max(dim=1, keepdim=True).values
        maturity_norm = (maturity_mat - vmin) / (vmax - vmin + 1e-8)
        maturity_norm = torch.clamp(maturity_norm, self.maturity_eps, 1.0)  # [L, T]

        # 存老师信息：每位老师的质心 + 基础 alpha + 每层成熟度（归一）
        self.teacher_info_list = []
        for t in range(T):
            layer_maturity_dict = {layer_keys[li]: float(maturity_norm[li, t].item()) for li in range(len(layer_keys))}
            self.teacher_info_list.append({
                "centroids": teacher_centroids_dicts[t],
                "alpha": float(alpha_base[t].item()),
                "layer_maturity": layer_maturity_dict
            })

    # -----------------------------
    # 对齐损失：KMeans + 教师混合质心重构
    # -----------------------------
    def _compute_alignment_loss(self) -> torch.Tensor:
        if (not self.is_align) or (len(self.teacher_info_list) == 0):
            return torch.zeros((), device=self.device)

        losses = []
        state = self.model.state_dict()
        device = self.device

        for layer_key in self.dkm_layers.keys():
            if not self.dkm_layers[layer_key].get("enabled", False):
                continue

            # 学生当前权重（展平）
            W = state[layer_key].to(device)
            Wf = W.view(-1, 1)

            # 老师的该层质心堆叠
            T = len(self.teacher_info_list)
            teacher_centroids = torch.stack(
                [torch.as_tensor(t['centroids'][layer_key], device=device) for t in self.teacher_info_list],
                dim=0
            )  # [T, K, 1]

            # 老师层面的权重：基础相似度 × 成熟度^gamma，再与基础平滑混合
            maturity_per_teacher = torch.tensor(
                [t['layer_maturity'].get(layer_key, 1.0) for t in self.teacher_info_list],
                device=device, dtype=torch.float
            )  # [T]
            alpha_base = torch.tensor([t['alpha'] for t in self.teacher_info_list], device=device, dtype=torch.float)  # [T]
            alpha_base_norm = alpha_base / (alpha_base.sum() + 1e-12)

            alpha_maturity = maturity_per_teacher ** self.teacher_gamma
            alpha_eff_raw = alpha_base * alpha_maturity
            if alpha_eff_raw.sum() <= 1e-12:
                alpha_eff_norm = torch.ones_like(alpha_eff_raw) / max(1, T)
            else:
                alpha_eff_norm = alpha_eff_raw / alpha_eff_raw.sum()
            alpha_eff = (1.0 - self.teacher_blend) * alpha_base_norm + self.teacher_blend * alpha_eff_norm
            alpha_eff = alpha_eff / (alpha_eff.sum() + 1e-12)

            # 学生侧：对当前 Wf 做一次 KMeans 得到 labels（轻量推断）
            with torch.no_grad():
                kmeans_s = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)
                kmeans_s.fit(Wf.detach())
                labels_s = kmeans_s.labels_.to(device)  # [N]

            # 教师混合质心：对每个簇 k，c_mix[k] = Σ_t alpha_eff[t] * teacher_centroids[t, k]
            c_mix = torch.einsum('t,tkc->kc', alpha_eff, teacher_centroids)  # -> [K, 1]

            # 用混合质心重构 X_rec，并计算层损失
            X_rec = c_mix[labels_s]  # [N,1]
            m_teacher_mix = (alpha_eff * maturity_per_teacher).sum().clamp_min(self.maturity_eps)
            loss_layer = m_teacher_mix * F.mse_loss(Wf, X_rec)
            losses.append(loss_layer)

        return torch.stack(losses).sum() if losses else torch.zeros((), device=device)

    # -----------------------------
    # 训练：先收老师 -> 本地训练 -> 聚类/成熟度 -> 清空邻居
    # -----------------------------
    def train(self):
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader 尚未初始化：请先在 Join 时调用 init_client()。")

        # 1) 基于邻居缓存构建老师信息（用于对齐损失）；此过程只读参数，不改参数
        self._all_teacher_info()

        # 2) 标准训练循环（无任何对参数的就地改动）
        self.model.train()
        for _ in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(x)
                loss_sup = self.criterion(outputs, labels).mean()

                if self.is_align and len(self.teacher_info_list) > 0:
                    loss_align = self._compute_alignment_loss()
                else:
                    loss_align = torch.zeros((), device=self.device)

                loss = loss_sup + self.lambda_alignment * loss_align
                loss.backward()
                self.optimizer.step()

        # 3) 训练结束后再做一次聚类/成熟度（只为通信构建 4 元组；不把 mask 应到参数）
        self.cluster_model = self._cluster_and_prune_model_weights()

        # 4) 本地通信轮 +1，清空邻居缓存
        self.local_round += 1
        self.neighbor_model_weights.clear()

    # -----------------------------
    # 发送：返回 4 元组（聚类状态 + 质心/标签 + 成熟度 meta）
    # -----------------------------
    def send_model(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        if self.cluster_model is None:
            self.cluster_model = self._cluster_and_prune_model_weights()
        clustered_state_dict, centroids_dict, labels_dict = self.cluster_model
        meta = self._prepare_maturity_meta()
        return clustered_state_dict, centroids_dict, labels_dict, meta

    # -----------------------------
    # 接收：只缓存老师，不做即时聚合
    # -----------------------------
    def receive_neighbor_model(self, neighbor_model):
        """
        期望邻居传来 3/4 元组（DFedMAC 表示）；也兼容 state_dict（将被忽略不用）。
        - 缓冲上限：buffer_limit；超过则丢弃最旧
        - 不触发 aggregate()
        """
        # 只接受 3/4 元组作为老师；若是纯 state_dict 则忽略（异步 DFedMAC 不用直接参数平均）
        if isinstance(neighbor_model, (tuple, list)) and (len(neighbor_model) in (3, 4)):
            self.neighbor_model_weights.append(neighbor_model)  # 缓存老师
            if self.buffer_limit and self.buffer_limit > 0 and len(self.neighbor_model_weights) > self.buffer_limit:
                overflow = len(self.neighbor_model_weights) - self.buffer_limit
                if overflow > 0:
                    self.neighbor_model_weights = self.neighbor_model_weights[overflow:]
        else:
            # 兼容性：收到非 DFedMAC 载荷则忽略
            pass

    # -----------------------------
    # 聚合：异步 DFedMAC 不做权重平均（No-Op）
    # -----------------------------
    def aggregate(self):
        """
        对齐型方法不做“收到即平均”，因此这里留空。
        若你想在某些实验中试验“参数平均 + 对齐”的混合策略，可在此读取
        self.neighbor_model_weights 里若存在 state_dict 的情况再做平均。
        """
        return

    # -----------------------------
    # Join 前设置初始模型
    # -----------------------------
    def set_init_model(self, model: torch.nn.Module):
        self.model = model.to(self.device)