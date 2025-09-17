from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, Any, Tuple, Optional, List
from copy import deepcopy

import torch
import torch.nn.functional as F

from clients.client import Client
from models.dkm import MultiTeacherDKMLayer
from utils.kmeans import TorchKMeans

def _cfd_distance(centroids_a: torch.Tensor,
                  centroids_b: torch.Tensor,
                  n_freqs: int = 512,
                  sigma: float = 1.0) -> float:
    """
    估计两组质心分布差异的距离，返回标量。
    """
    device = centroids_a.device
    if centroids_a.ndim == 1:
        centroids_a = centroids_a.view(-1, 1)
        centroids_b = centroids_b.view(-1, 1)
    D = centroids_a.shape[1]
    freqs = torch.randn(n_freqs, D, device=device) * sigma
    fa = (freqs @ centroids_a.T)   # [n_freqs, K]
    fb = (freqs @ centroids_b.T)
    phi_a = torch.mean(torch.exp(1j * fa), dim=1)  # [n_freqs]
    phi_b = torch.mean(torch.exp(1j * fb), dim=1)
    cfd = torch.mean(torch.abs(phi_a - phi_b) ** 2)
    return cfd.item() if not isinstance(cfd, float) else cfd


class ADFedMACClient(Client):
    """
    Asynchronous Decentralized FedMAC（成熟度感知的结构对齐，异步事件友好版）

    事件语义（由 AsyncCoordinator 驱动）：
      - JOIN: 协调器先 set_init_model() 再 init_client()，随后用 compute_time_for_next_burst() 估算首个 TRAIN_DONE 时间。
      - TRAIN_DONE: 协调器调用 client.train()；client 内部：
          1) 读取邻居缓冲，构建老师集合（含成熟度与 CFD）；
          2) 本地训练（监督 + 成熟度加权的结构对齐）；
          3) 训练后做聚类/剪枝，更新本地历史，生成 maturity meta；
          4) 清空邻居缓冲（等待下一次接收）。
        协调器随后调用 client.send_model() 获取 4 元组载荷并向在线邻居转发。
      - RECEIVE: 协调器把发送方的 4 元组交给接收方的 receive_neighbor_model()；本类默认仅缓冲，不立刻训练。
    """

    # ===================== 初始化 & 超参 =====================
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        # 重要：默认在异步对齐中，收到不立刻融合（缓冲到下一次训练使用）
        hyperparam = dict(hyperparam)
        hyperparam.setdefault('fuse_on_receive', False)

        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        # --- 对齐/结构化通信超参 ---
        self.lambda_alignment: float = float(hyperparam.get('lambda_alignment', 0.1))
        self.n_clusters: int = int(hyperparam.get('n_clusters', 16))

        # 教师柔性降权（使用成熟度）——与“DFedMAC”一致
        self.teacher_gamma: float = float(hyperparam.get('teacher_gamma', 1.0))
        self.teacher_blend: float = float(hyperparam.get('teacher_blend', 0.6))  # 0仅相似度；1仅成熟度重标定

        # 成熟度（C-SWAG × 稳定度）相关
        self.maturity_window: int = int(hyperparam.get('maturity_window', 5))
        self.beta_drift: float = float(hyperparam.get('beta_drift', 1.0))
        self.beta_invar: float = float(hyperparam.get('beta_invar', 1.0))
        self.beta_mask: float = float(hyperparam.get('beta_mask', 0.5))
        self.maturity_eps: float = float(hyperparam.get('maturity_eps', 1e-8))

        # 运行时容器
        self.teacher_info_list: List[Dict[str, Any]] = []   # 每位老师：{'centroids', 'alpha', 'layer_maturity'}
        self.dkm_layers: Dict[str, MultiTeacherDKMLayer] = {}
        self.mask: Dict[str, torch.Tensor] = {}

        # 聚类缓存（训练结束后生成，send_model 复用，避免重复 KMeans）
        self.cluster_model: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None

        # 本地历史（用于成熟度）
        self._local_hist = defaultdict(lambda: deque(maxlen=self.maturity_window))  # {layer: [K×1 tensor]}
        self._local_mask_hist = defaultdict(lambda: deque(maxlen=2))               # {layer: [mask tensor(bool)]}

        # 标志：是否启用对齐（邻居为空或 λ=0 则训练仅用监督损失）
        self.is_align: bool = (self.lambda_alignment != 0.0)

    # ===================== 工具：质心排序/重映射 & 簇内方差 =====================
    @staticmethod
    def _sort_centroids_and_remap(centroids_1d: torch.Tensor, labels_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按值对质心排序，并把标签 remap 到排序后的索引，增强跨轮配对稳定性。
        """
        c = centroids_1d.view(-1).detach()
        order = torch.argsort(c)  # 升序
        sorted_c = c[order].view(-1, 1)
        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        new_labels = old2new[labels_1d]
        return sorted_c, new_labels

    @staticmethod
    def _cluster_intra_var(flat_weights: torch.Tensor, labels: torch.Tensor, K: int, eps: float = 1e-8) -> torch.Tensor:
        """
        逐簇计算当前层的簇内方差（[K,1]）。
        """
        vars_out = flat_weights.new_zeros(K, 1)
        for k in range(K):
            idx = (labels == k)
            if idx.any():
                w = flat_weights[idx].view(-1)
                vars_out[k, 0] = torch.var(w, unbiased=False)
            else:
                vars_out[k, 0] = eps
        return vars_out + eps

    # ===================== 历史维护（成熟度要用） =====================
    def _update_local_history(self, centroids_dict: Dict[str, torch.Tensor], mask_dict: Dict[str, torch.Tensor]) -> None:
        for layer, c in centroids_dict.items():
            self._local_hist[layer].append(c.detach().cpu())
        for layer, m in mask_dict.items():
            self._local_mask_hist[layer].append(m.detach().cpu())

    # ===================== 注册 DKM 层（与 DFedCAD 风格一致） =====================
    def _register_dkm_layers(self) -> None:
        self.dkm_layers = {}
        for key in self.model.state_dict().keys():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key and 'conv' in key:
                self.dkm_layers[key] = MultiTeacherDKMLayer(
                    n_clusters=self.n_clusters,
                    alpha_mix=0.7,  # 与 DFedCAD 一致
                    beta_dist=2.0
                ).to(self.device)

    # ===================== 剪枝（按 mask） =====================
    def _prune_model_weights(self) -> Dict[str, torch.Tensor]:
        pruned = {}
        for key, weight in self.model.state_dict().items():
            if key in self.mask:
                pruned[key] = weight * self.mask[key]
            else:
                pruned[key] = weight
        return pruned

    # ===================== 聚类 + 剪枝 + 质心/标签（带排序重映射） =====================
    def _cluster_and_prune_model_weights(self) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        返回：
          clustered_state_dict:  聚类重构后的参数（保留原形状）
          centroids_dict:        每层 K×1 的质心张量
          labels_dict:           每层展平后的标签（LongTensor [N]）
        同时更新 self.mask & 本地历史（用于成熟度）。
        """
        clustered_state_dict: Dict[str, torch.Tensor] = {}
        mask_dict: Dict[str, torch.Tensor] = {}
        centroids_dict: Dict[str, torch.Tensor] = {}
        labels_dict: Dict[str, torch.Tensor] = {}

        state = self.model.state_dict()
        for key, weight in state.items():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key:
                original_shape = weight.shape
                kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)
                flat = weight.detach().view(-1, 1)
                kmeans.fit(flat)  # 产生 centroids[K,1], labels_[N]

                # 质心排序 + 标签重映射
                centroids_sorted, labels_sorted = self._sort_centroids_and_remap(
                    kmeans.centroids.view(-1, 1), kmeans.labels_
                )

                # 重构 & mask（0 质心视为剪枝）
                new_weights = centroids_sorted[labels_sorted].view(original_shape)
                is_zero = (centroids_sorted.view(-1) == 0)
                mask = (is_zero[labels_sorted].view(original_shape) == 0)

                clustered_state_dict[key] = new_weights
                mask_dict[key] = mask.bool()
                centroids_dict[key] = centroids_sorted.to(self.device)
                labels_dict[key] = labels_sorted.view(-1).to(self.device)
            else:
                clustered_state_dict[key] = weight
                mask_dict[key] = torch.ones_like(weight, dtype=torch.bool)

        # 更新本地历史/掩码
        self.mask = mask_dict
        self._update_local_history(centroids_dict, mask_dict)

        return clustered_state_dict, centroids_dict, labels_dict

    # ===================== 成熟度（C-SWAG × 稳定度） =====================
    def _c_swag_precision(self, hist_deque: deque) -> torch.Tensor:
        if len(hist_deque) < 2:
            k = hist_deque[-1].shape[0]
            return hist_deque[-1].new_ones(k, 1)
        stack = torch.stack(list(hist_deque), dim=0)       # [T,K,1]
        var = torch.var(stack, dim=0, unbiased=False)      # [K,1]
        return 1.0 / (var + self.maturity_eps)

    def _stability_scores(self, layer_key: str,
                          centroids_now: torch.Tensor,
                          labels_now: torch.Tensor,
                          mask_now: torch.Tensor) -> torch.Tensor:
        device = centroids_now.device
        # 漂移：与上一帧质心差
        if len(self._local_hist[layer_key]) >= 1:
            prev_c = self._local_hist[layer_key][-1].to(device)
            drift = torch.abs(centroids_now - prev_c)
        else:
            drift = torch.zeros_like(centroids_now)

        # 簇内方差（当前）
        flat_w = self.model.state_dict()[layer_key].to(device).view(-1, 1).detach()
        K = centroids_now.shape[0]
        invar = self._cluster_intra_var(flat_w, labels_now, K, eps=self.maturity_eps)

        # mask 翻转率（层级标量）
        if len(self._local_mask_hist[layer_key]) >= 1:
            prev_m = self._local_mask_hist[layer_key][-1].to(device)
            flips = (prev_m ^ mask_now).float().mean()
        else:
            flips = torch.tensor(0.0, device=device)

        stab = torch.exp(
            - self.beta_drift * drift
            - self.beta_invar * invar
            - self.beta_mask * flips
        )
        return stab.clamp_min(1e-6)

    def _local_maturity(self,
                        layer_key: str,
                        centroids_now: torch.Tensor,
                        labels_now: torch.Tensor,
                        mask_now: torch.Tensor) -> torch.Tensor:
        lam = self._c_swag_precision(self._local_hist[layer_key]).to(centroids_now.device)
        stab = self._stability_scores(layer_key, centroids_now, labels_now, mask_now)
        return lam * stab  # [K,1]

    def _prepare_maturity_meta(self,
                               centroids_dict: Dict[str, torch.Tensor],
                               labels_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        使用“当前（刚聚类）的质心/标签/掩码 + 本地历史”计算层级成熟度标量，并封装 meta。
        """
        layer_maturity = {}
        for layer_key, centroids_now in centroids_dict.items():
            labels_now = labels_dict[layer_key]
            mask_now = self.mask[layer_key].to(centroids_now.device)
            maturity_vec = self._local_maturity(layer_key, centroids_now, labels_now, mask_now)
            layer_maturity[layer_key] = float(maturity_vec.mean().item())

        meta = {
            'version_meta': 'dfedmac_meta_v1',
            'sender_id': self.id,
            'version': int(self.local_version),     # 从基类维护的本地版本（train_done 后由协调器 on_train_done）
            'sender_time': float(self.last_update_time),
            'layer_maturity': layer_maturity
        }
        return meta

    # ===================== 教师集合（CFD + 成熟度） =====================
    def _all_teacher_info(self) -> None:
        """
        构造教师集合：
          - 先用本地模型做一次聚类（仅获取本地质心；不持久化为 cluster_model）
          - 遍历邻居缓冲（3元：Wc, C, L；4元：Wc, C, L, meta），为每位老师计算:
              * 与本地的 CFD 相似度（分层平均）→ 基础权重 alpha_base（softmax）
              * 从 meta 取该层成熟度，做“层内 min-max 归一”
          - 得到 self.teacher_info_list: [{'centroids':dict, 'alpha':float, 'layer_maturity':{...}}, ...]
        """
        # 本地质心（临时）
        _, local_centroids_dict, _ = self._cluster_and_prune_model_weights()

        # 收集老师
        teacher_centroids_dicts: List[Dict[str, torch.Tensor]] = []
        teacher_meta_list: List[Optional[Dict[str, Any]]] = []
        cfd_matrix: List[List[float]] = []

        for item in self.neighbor_model_weights:
            # 兼容 3/4 元组
            if isinstance(item, (list, tuple)) and (len(item) == 4):
                _, teacher_centroids, _, meta = item
                teacher_meta_list.append(meta)
            elif isinstance(item, (list, tuple)) and (len(item) == 3):
                _, teacher_centroids, _ = item
                meta = None
                teacher_meta_list.append(meta)
            else:
                # 其它类型（比如误发 state_dict）直接跳过
                continue

            teacher_centroids_dicts.append(teacher_centroids)

            per_layer = []
            for layer_key in local_centroids_dict:
                cfd = _cfd_distance(
                    local_centroids_dict[layer_key].detach().float(),
                    teacher_centroids[layer_key].detach().float()
                )
                per_layer.append(cfd)
            cfd_matrix.append(per_layer)

        if len(teacher_centroids_dicts) == 0:
            self.teacher_info_list = []
            return

        cfd_tensor = torch.tensor(cfd_matrix, dtype=torch.float, device=self.device)  # [T, L]
        cfd_scores = torch.mean(cfd_tensor, dim=1)  # [T]

        # 基础相似度权重（CFD 小→更相似→更高权）
        min_val, max_val = cfd_scores.min(), cfd_scores.max()
        normed = (cfd_scores - min_val) / (max_val - min_val + 1e-8)
        beta = 2.0
        alpha_base = torch.softmax(-beta * normed, dim=0)  # [T]

        # 层级成熟度（若老师未提供，则当 1.0）
        layer_keys = list(local_centroids_dict.keys())
        T = len(teacher_centroids_dicts)
        maturity_mat = torch.ones(len(layer_keys), T, dtype=torch.float, device=self.device)
        for t_idx, meta in enumerate(teacher_meta_list):
            if isinstance(meta, dict) and ('layer_maturity' in meta):
                lm = meta['layer_maturity']
                for li, layer_key in enumerate(layer_keys):
                    if layer_key in lm:
                        maturity_mat[li, t_idx] = max(float(lm[layer_key]), self.maturity_eps)

        # 对每层做 min-max 归一（避免某位老师的绝对标量全局偏大/偏小）
        vmin = maturity_mat.min(dim=1, keepdim=True).values
        vmax = maturity_mat.max(dim=1, keepdim=True).values
        maturity_norm = (maturity_mat - vmin) / (vmax - vmin + 1e-8)
        maturity_norm = torch.clamp(maturity_norm, self.maturity_eps, 1.0)  # [L, T]

        # 打包
        self.teacher_info_list = []
        for t in range(T):
            layer_maturity_dict = {layer_keys[li]: float(maturity_norm[li, t].item())
                                   for li in range(len(layer_keys))}
            self.teacher_info_list.append({
                'centroids': teacher_centroids_dicts[t],
                'alpha': float(alpha_base[t].item()),
                'layer_maturity': layer_maturity_dict
            })

    # ===================== 对齐损失（成熟度感知的教师权重 + 层级缩放） =====================
    def _compute_alignment_loss(self) -> torch.Tensor:
        if (not self.is_align) or len(self.teacher_info_list) == 0 or len(self.dkm_layers) == 0:
            return torch.zeros((), device=self.device)

        losses = []
        state = self.model.state_dict()

        for layer_key, dkm in self.dkm_layers.items():
            W = state[layer_key].to(self.device)
            Wf = W.view(-1, 1)

            # 堆叠教师该层质心
            teacher_centroids = torch.stack(
                [t['centroids'][layer_key].to(self.device) for t in self.teacher_info_list],
                dim=0  # [T,K,1]
            )

            # 教师权重：基础相似度 * 成熟度^gamma（并归一），再与基础权重按 teacher_blend 融合
            maturity_per_teacher = torch.tensor(
                [t['layer_maturity'].get(layer_key, 1.0) for t in self.teacher_info_list],
                device=self.device, dtype=torch.float
            )  # [T]

            alpha_base = torch.tensor(
                [t['alpha'] for t in self.teacher_info_list],
                device=self.device, dtype=torch.float
            )  # [T]
            alpha_base_norm = alpha_base / (alpha_base.sum() + 1e-12)

            alpha_maturity = maturity_per_teacher ** self.teacher_gamma
            alpha_eff_raw = alpha_base * alpha_maturity
            if alpha_eff_raw.sum() <= 1e-12:
                alpha_eff_norm = torch.ones_like(alpha_eff_raw) / max(1, alpha_eff_raw.numel())
            else:
                alpha_eff_norm = alpha_eff_raw / alpha_eff_raw.sum()

            alpha_eff = (1.0 - self.teacher_blend) * alpha_base_norm + self.teacher_blend * alpha_eff_norm
            alpha_eff = alpha_eff / (alpha_eff.sum() + 1e-12)

            # DKM 重构
            X_rec, _, _ = dkm(
                Wf,
                teacher_centroids=teacher_centroids,
                teacher_alphas=alpha_eff,
                teacher_index_tables=None,
                lambda_teacher=self.lambda_alignment
            )

            # 层级缩放（成熟度×权重混合）
            m_teacher_mix = (alpha_eff * maturity_per_teacher).sum().clamp_min(self.maturity_eps)
            loss_layer = m_teacher_mix * F.mse_loss(Wf, X_rec)
            losses.append(loss_layer)

        return torch.stack(losses).sum() if losses else torch.zeros((), device=self.device)

    # ===================== 覆盖：接收邻居（缓冲，不立即融合） =====================
    def receive_neighbor_model(self, neighbor_model: Any):
        """
        期望接收 3/4 元组：
          (clustered_state_dict, centroids_dict, labels_dict [, meta])
        收到后仅进入缓冲；在下一次 train() 前被 _all_teacher_info() 使用。
        """
        if isinstance(neighbor_model, (list, tuple)) and (len(neighbor_model) in (3, 4)):
            self.neighbor_model_weights.append(neighbor_model)
        else:
            # 若误传 state_dict（比如把异步 FedAvg 的载荷发到了这里），直接丢弃，避免污染
            return

        # 控制缓冲大小
        if self.buffer_limit is not None and self.buffer_limit > 0:
            overflow = len(self.neighbor_model_weights) - self.buffer_limit
            if overflow > 0:
                self.neighbor_model_weights = self.neighbor_model_weights[overflow:]

        # ADFedMAC 不做“收到即融合”，忽略 fuse_on_receive

    # ===================== 覆盖：聚合（用于收到即融合的场景，这里空实现以防误调） =====================
    def aggregate(self):
        """
        ADFedMAC 不对权重做直接平均聚合；此处留空（若外部误调用 fuse_on_receive，也不会改变模型）。
        """
        return

    # ===================== 训练流程（异步友好） =====================
    def train(self):
        """
        一次“本地训练单元（burst）”：
          - 根据邻居缓冲构建教师集合（含 CFD 与成熟度）
          - 逐 batch：剪枝、监督损失 + 成熟度感知的结构对齐损失
          - 训练后：聚类/剪枝，更新历史，缓存 cluster_model 并清空邻居缓冲
        """
        # 注册 DKM 层
        if self.is_align and len(self.dkm_layers) == 0:
            self._register_dkm_layers()

        # 先准备教师集合（若无邻居则 teacher_info_list 为空，等价于无对齐）
        if self.is_align:
            self._all_teacher_info()

        # 本地训练
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader has not been initialized. Please ensure that init_client() is called when joining.")

        self.model.train()
        for _ in range(self.epochs):
            for x, labels in self.client_train_loader:
                # 按当前 mask 施加结构
                self.model.load_state_dict(self._prune_model_weights())

                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x)
                loss_sup = self.criterion(outputs, labels).mean()

                loss_align = self._compute_alignment_loss() if self.is_align else torch.zeros((), device=self.device)
                loss_final = loss_sup + self.lambda_alignment * loss_align
                loss_final.backward()
                self.optimizer.step()

        # 训练后，准备聚类表示（供 send_model 使用），并更新本地历史
        self.cluster_model = self._cluster_and_prune_model_weights()

        # 清空邻居缓冲（下个 burst 再收）
        self.neighbor_model_weights.clear()

    # ===================== 初始化模型 =====================
    def set_init_model(self, model: torch.nn.Module):
        """
        直接深拷贝传入的模型实例，避免重新构造导致的构造参数缺失问题（如 LeNet 需要 num_classes）。
        """
        self.model = deepcopy(model)
        self.model.to(self.device)
        # 清理/复位与结构化通信相关的缓存
        self.cluster_model = None
        self.teacher_info_list.clear()
        self.mask.clear()

    # ===================== 发送（4 元组：聚类重构 + 质心/标签 + 成熟度 meta） =====================
    def send_model(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        返回 4 元组（与 DFedMAC/DFedCAD 的结构化通信兼容）：
          (clustered_state_dict, centroids_dict, labels_dict, meta)
        其中 meta['layer_maturity'] 为每层的成熟度标量，来自刚完成的聚类结果 + 历史统计。
        """
        if self.cluster_model is None:
            # 若外部在没有训练的情况下调用，临时做一次聚类（不修改历史）
            clustered_state_dict, centroids_dict, labels_dict = self._cluster_and_prune_model_weights()
        else:
            clustered_state_dict, centroids_dict, labels_dict = self.cluster_model

        maturity_meta = self._prepare_maturity_meta(centroids_dict, labels_dict)
        return clustered_state_dict, centroids_dict, labels_dict, maturity_meta