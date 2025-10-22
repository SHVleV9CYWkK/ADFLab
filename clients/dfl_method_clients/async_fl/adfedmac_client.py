from __future__ import annotations
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from clients.client import Client
from models.dkm import MultiTeacherDKMLayer
from utils.kmeans import TorchKMeans


# --------------------------- 工具：CFD 距离 ---------------------------
def _cfd_distance(centroids_a: torch.Tensor,
                  centroids_b: torch.Tensor,
                  n_freqs: int = 512,
                  sigma: float = 1.0) -> float:
    device = centroids_a.device
    if centroids_a.ndim == 1:
        centroids_a = centroids_a.view(-1, 1)
        centroids_b = centroids_b.view(-1, 1)
    D = centroids_a.shape[1]
    freqs = torch.randn(n_freqs, D, device=device) * sigma
    fa = (freqs @ centroids_a.T)
    fb = (freqs @ centroids_b.T)
    phi_a = torch.mean(torch.exp(1j * fa), dim=1)
    phi_b = torch.mean(torch.exp(1j * fb), dim=1)
    cfd = torch.mean(torch.abs(phi_a - phi_b) ** 2)
    return cfd.item() if not isinstance(cfd, float) else cfd


class ADFedMACClient(Client):
    """
    混合策略（不修改协调器、无补发）：

    - 普通客户端（非延迟）：
        * 缓冲邻居的 state_dict；
        * 训练前（或 set_init_model 当刻）做一次等权平均： self + neighbors；
        * 训练使用基类 _local_train()；
        * 不进行结构对齐。
    - 延迟客户端：
        * 预训练阶段：行为与上面“普通客户端”一致（可用等权聚合 + _local_train()），不做对齐；
        * 预训练结束：切换到结构对齐模式（DKM + 成熟度/CFD），此后不再做直接平均聚合。
    """

    # ===================== 初始化 & 策略开关 =====================
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        hp = dict(hyperparam)

        self.is_delayed: bool = hp.get('is_delayed', False)

        # 预训练（延迟客户端专用；单位=burst）
        self.warmup_bursts: int = int(hp.get('warmup_bursts', 1))
        self._warmup_done: int = 0

        # 与 AsyncDFedAvgClient 一致：不强制“收到即融合”，聚合时机由上层调度或 train() 前触发
        self.fuse_on_receive: bool = bool(hp.get('fuse_on_receive', False))

        # 对齐相关（仅延迟客户端在预训练后启用）
        self.lambda_alignment: float = float(hp.get('lambda_alignment', 0.01))
        self.n_clusters: int = int(hp.get('n_clusters', 16))
        self.base_decay_rate = hyperparam.get('base_decay_rate', 0.5)
        self.teacher_gamma: float = float(hp.get('teacher_gamma', 1.0))
        self.teacher_blend: float = float(hp.get('teacher_blend', 0.6))
        self.maturity_window: int = int(hp.get('maturity_window', 5))
        self.beta_drift: float = float(hp.get('beta_drift', 1.0))
        self.beta_invar: float = float(hp.get('beta_invar', 1.0))
        self.beta_mask: float = float(hp.get('beta_mask', 0.5))
        self.maturity_eps: float = float(hp.get('maturity_eps', 1e-8))

        super().__init__(client_id, dataset_index, full_dataset, hp, device)

        # 用于“对齐”的邻居缓存（存 3/4 元组：(clustered_state, centroids, labels [,meta])），仅延迟-对齐使用
        self.neighbor_model_weights: List[Any] = []

        # 对齐组件
        self.teacher_info_list: List[Dict[str, Any]] = []
        self.dkm_layers: Dict[str, MultiTeacherDKMLayer] = {}
        self.mask: Dict[str, torch.Tensor] = {}
        self._local_hist = defaultdict(lambda: deque(maxlen=self.maturity_window))
        self._local_mask_hist = defaultdict(lambda: deque(maxlen=2))

        # 训练后缓存（避免重复聚类）
        self.cluster_model: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None

        self.global_model = None

    def _compute_global_local_model_difference(self):
        global_dict = self.global_model.state_dict()
        local_dict = self.model.state_dict()
        difference_dict = {}
        for key in global_dict:
            difference_dict[key] = local_dict[key] - global_dict[key]
        return difference_dict

    # ===================== DKM/聚类/成熟度（与前版一致，略微精简） =====================
    def _register_dkm_layers(self):
        if self.lambda_alignment == 0.0:
            return

        self.dkm_layers = {}
        for key in self.model.state_dict().keys():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key and 'conv' in key:
                self.dkm_layers[key] = MultiTeacherDKMLayer(
                    n_clusters=self.n_clusters, alpha_mix=0.7, beta_dist=2.0
                ).to(self.device)

    def _prune_model_weights(self):
        pruned_state_dict = {}
        for key, weight in self.model.state_dict().items():
            if key in self.mask:
                pruned_state_dict[key] = weight * self.mask[key]
            else:
                pruned_state_dict[key] = weight
        return pruned_state_dict

    @staticmethod
    def _sort_centroids_and_remap(c1d: torch.Tensor, lbl: torch.Tensor):
        c = c1d.view(-1).detach()
        order = torch.argsort(c)
        sorted_c = c[order].view(-1, 1)
        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        new_labels = old2new[lbl]
        return sorted_c, new_labels

    @staticmethod
    def _cluster_intra_var(flat_w: torch.Tensor, labels: torch.Tensor, K: int, eps: float = 1e-8) -> torch.Tensor:
        vars_out = flat_w.new_zeros(K, 1)
        for k in range(K):
            idx = (labels == k)
            if idx.any():
                w = flat_w[idx].view(-1)
                vars_out[k, 0] = torch.var(w, unbiased=False)
            else:
                vars_out[k, 0] = eps
        return vars_out + eps

    def _update_local_history(self, centroids: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]):
        for layer, c in centroids.items():
            self._local_hist[layer].append(c.detach())
        for layer, m in mask.items():
            self._local_mask_hist[layer].append(m.detach())

    def _cluster_and_prune_model_weights(self):
        clustered: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}
        cents: Dict[str, torch.Tensor] = {}
        labels: Dict[str, torch.Tensor] = {}

        state = self.model.state_dict()
        for key, w in state.items():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key:
                orig = w.shape
                kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)
                flat = w.detach().view(-1, 1)
                kmeans.fit(flat)
                cent_sorted, lab_sorted = self._sort_centroids_and_remap(
                    kmeans.centroids.view(-1, 1), kmeans.labels_)
                new_w = cent_sorted[lab_sorted].view(orig)
                is_zero = (cent_sorted.view(-1) == 0)
                m = (is_zero[lab_sorted].view(orig) == 0)

                clustered[key] = new_w
                mask[key] = m.bool()
                cents[key] = cent_sorted.to(self.device)
                labels[key] = lab_sorted.view(-1).to(self.device)
            else:
                clustered[key] = w
                mask[key] = torch.ones_like(w, dtype=torch.bool)

        self.mask = mask
        self._update_local_history(cents, mask)
        return clustered, cents, labels

    def _c_swag_precision(self, hist: deque) -> torch.Tensor:
        last = hist[-1]
        if len(hist) < 2:
            return last.new_ones(last.shape[0], 1)

        stack = torch.stack(list(hist), dim=0)  # [T,K,1]
        mu = stack.mean(dim=0)  # [K,1]
        var = ((stack - mu) ** 2).mean(dim=0)  # [K,1]

        # 相对方差（对尺度鲁棒）
        cv = var / (mu.abs() + 1e-4)  # [K,1]
        tau = getattr(self, "maturity_tau", 10.0)  # 超参：越大越挑剔
        lam = torch.exp(-tau * cv).clamp_min(self.maturity_eps)  # (eps,1]
        return lam

    def _stability_scores(self, layer_key: str,
                          cents_now: torch.Tensor,
                          labels_now: torch.Tensor,
                          mask_now: torch.Tensor) -> torch.Tensor:
        dev = cents_now.device
        # 漂移：和上一帧质心差
        if len(self._local_hist[layer_key]) >= 1:
            prev_c = self._local_hist[layer_key][-1].to(dev)
            drift = torch.abs(cents_now - prev_c)  # [K,1]
        else:
            drift = torch.zeros_like(cents_now)

        # 簇内方差（当前帧；若担心开销，可对 flat_w 子采样）
        flat_w = self.model.state_dict()[layer_key].to(dev).view(-1, 1).detach()
        K = cents_now.shape[0]
        invar = self._cluster_intra_var(flat_w, labels_now, K, eps=self.maturity_eps)  # [K,1]

        # mask 翻转率（层级标量）
        if len(self._local_mask_hist[layer_key]) >= 1:
            prev_m = self._local_mask_hist[layer_key][-1].to(dev)
            flips = (prev_m ^ mask_now).float().mean()  # scalar
        else:
            flips = torch.tensor(0.0, device=dev)

        stab = torch.exp(
            - self.beta_drift * drift
            - self.beta_invar * invar
            - self.beta_mask * flips
        )
        return stab

    def _local_maturity(self, layer_key: str,
                        cents_now: torch.Tensor,
                        labels_now: torch.Tensor,
                        mask_now: torch.Tensor) -> torch.Tensor:
        lam = self._c_swag_precision(self._local_hist[layer_key]).to(cents_now.device)
        stab = self._stability_scores(layer_key, cents_now, labels_now, mask_now)
        return lam * stab

    def _prepare_maturity_meta(self, cents: Dict[str, torch.Tensor],
                               labels: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # layer_maturity = {}
        # for layer_key, c_now in cents.items():
        #     m_now = self.mask[layer_key].to(c_now.device)
        #     l_now = labels[layer_key]
        #     maturity_vec = self._local_maturity(layer_key, c_now, l_now, m_now)
        #     layer_maturity[layer_key] = float(maturity_vec.mean().item())
        return {
            'version_meta': 'dfedmac_meta_v1',
            'sender_id': self.id,
            'version': int(self.local_version),
            'sender_time': float(self.last_update_time),
            # 'layer_maturity': layer_maturity
        }

    def _all_teacher_info(self) -> None:
        if self.cluster_model is None or self.lambda_alignment == 0.0:
            return
        else:
            local_cents = self.cluster_model[1]

        t_cents_list: List[Dict[str, torch.Tensor]] = []
        t_meta_list: List[Optional[Dict[str, Any]]] = []
        cfd_matrix: List[List[float]] = []

        for item in self.neighbor_model_weights:
            _, c_t, _, meta = item
            t_meta_list.append(meta)

            t_cents_list.append(c_t)

            per_layer = []
            for lk in local_cents:
                per_layer.append(_cfd_distance(local_cents[lk].detach().float(),
                                               c_t[lk].detach().float()))
            cfd_matrix.append(per_layer)

        if len(t_cents_list) == 0:
            self.teacher_info_list = []
            return

        cfd_tensor = torch.tensor(cfd_matrix, dtype=torch.float, device=self.device)  # [T,L]
        cfd_scores = torch.mean(cfd_tensor, dim=1)  # [T]
        min_v, max_v = cfd_scores.min(), cfd_scores.max()
        normed = (cfd_scores - min_v) / (max_v - min_v + 1e-8)
        alpha_base = torch.softmax(-2.0 * normed, dim=0)

        layer_keys = list(local_cents.keys())
        T = len(t_cents_list)
        # maturity_mat = torch.ones(len(layer_keys), T, dtype=torch.float, device=self.device)
        # for t_idx, meta in enumerate(t_meta_list):
        #     if isinstance(meta, dict) and ('layer_maturity' in meta):
        #         lm = meta['layer_maturity']
        #         for li, lk in enumerate(layer_keys):
        #             if lk in lm:
        #                 maturity_mat[li, t_idx] = max(float(lm[lk]), self.maturity_eps)
        #
        # vmin = maturity_mat.min(dim=1, keepdim=True).values
        # vmax = maturity_mat.max(dim=1, keepdim=True).values
        # maturity_norm = torch.clamp((maturity_mat - vmin) / (vmax - vmin + 1e-8),
        #                             self.maturity_eps, 1.0)

        self.teacher_info_list = []
        for t in range(T):
            # layer_maturity_dict = {layer_keys[li]: float(maturity_norm[li, t].item())
            #                        for li in range(len(layer_keys))}
            self.teacher_info_list.append({
                'centroids': t_cents_list[t],
                'alpha': float(alpha_base[t].item()),
                # 'layer_maturity': layer_maturity_dict
            })

    def _compute_alignment_loss(self) -> torch.Tensor:
        # if len(self.teacher_info_list) == 0 or len(self.dkm_layers) == 0 or self.lambda_alignment == 0.0:
        #     return torch.zeros((), device=self.device)
        #
        # losses = []
        # st = self.model.state_dict()
        # for lk, dkm in self.dkm_layers.items():
        #     Wf = st[lk].to(self.device).view(-1, 1)
        #     t_cents = torch.stack([t['centroids'][lk].to(self.device)
        #                            for t in self.teacher_info_list], dim=0)
        #     maturity = torch.tensor([t['layer_maturity'].get(lk, 0.0)
        #                              for t in self.teacher_info_list],
        #                             device=self.device, dtype=torch.float)
        #     alpha_base = torch.tensor([t['alpha'] for t in self.teacher_info_list],
        #                               device=self.device, dtype=torch.float)
        #     alpha_base = alpha_base / (alpha_base.sum() + 1e-12)
        #
        #     alpha_eff_raw = alpha_base * (maturity ** self.teacher_gamma)
        #     alpha_eff = (alpha_eff_raw / alpha_eff_raw.sum()
        #                  if alpha_eff_raw.sum() > 1e-12
        #                  else torch.ones_like(alpha_eff_raw) / max(1, alpha_eff_raw.numel()))
        #
        #     alpha_eff = (1.0 - self.teacher_blend) * alpha_base + self.teacher_blend * alpha_eff
        #     alpha_eff = alpha_eff / (alpha_eff.sum() + 1e-12)
        #
        #     X_rec, _, _ = dkm(
        #         Wf,
        #         teacher_centroids=t_cents,
        #         teacher_alphas=alpha_eff,
        #         teacher_index_tables=None,
        #         lambda_teacher=1.0
        #     )
        #     losses.append(F.mse_loss(Wf, X_rec))
        #
        # return torch.stack(losses).sum() if losses else torch.zeros((), device=self.device)

        if len(self.teacher_info_list) == 0 or self.lambda_alignment == 0.0:
            return torch.zeros((), device=self.device)

        losses = []
        # 多教师质心 & labels 都在 self.teacher_info_list
        for layer_key, dkm in self.dkm_layers.items():
            # 1) 拿学生当前权重
            param = dict(self.model.named_parameters())[layer_key]
            Wf = param.view(-1, 1)  # (N_w,1)

            # 2) 准备教师输入
            teacher_centroids = torch.stack(
                [t["centroids"][layer_key].to(self.device)
                 for t in self.teacher_info_list], dim=0  # (T,K,1)
            )
            teacher_alphas = torch.tensor([t["alpha"] for t in self.teacher_info_list], device=self.device)  # (T,)

            # 3) 调用 DKM 层，得到重构
            X_rec, _, _ = dkm(
                Wf,
                teacher_centroids=teacher_centroids,
                teacher_alphas=teacher_alphas,
                teacher_index_tables=None
            )

            # 4) 重构误差
            Wc = Wf - Wf.mean(dim=0, keepdim=True)
            losses.append(F.mse_loss(Wc, X_rec, reduction="mean") / (Wf.var() + 1e-8))

        if losses:
            return torch.stack(losses).sum()
        else:
            return torch.zeros((), device=self.device)

    def _local_train(self, is_lambda=False):
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader not initialized; ensure init_client() on join.")

        ref_momentum = self._compute_global_local_model_difference()

        self.model.train()

        exponential_average_loss = None
        alpha = 0.5

        for _ in range(self.epochs):
            for batch_idx, (x, labels) in enumerate(self.client_train_loader):
                self.model.load_state_dict(self._prune_model_weights())
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x)
                loss_sup = self.criterion(outputs, labels).mean()

                loss_align = self._compute_alignment_loss()

                loss_final = loss_sup + self.lambda_alignment * loss_align
                loss_final.backward()

                if exponential_average_loss is None:
                    exponential_average_loss = loss_final.item()
                else:
                    exponential_average_loss = alpha * loss_final.item() + (1 - alpha) * exponential_average_loss

                if loss_final.item() < exponential_average_loss:
                    decay_factor = min(self.base_decay_rate ** (batch_idx + 1) * 1.1, 0.8)
                else:
                    decay_factor = max(self.base_decay_rate ** (batch_idx + 1) / 1.1, 0.1)

                for name, param in self.model.named_parameters():
                    if name in ref_momentum:
                        param.grad += decay_factor * ref_momentum[name]

                self.optimizer.step()

    @torch.no_grad()
    def aggregate(self):
        neighbor_weights_state = [neighbor[0] for neighbor in self.neighbor_model_weights]
        if len(neighbor_weights_state) == 0:
            return  # 没有邻居更新就不动

        average_weights = {}
        for key in neighbor_weights_state[0].keys():
            weighted_sum = sum(
                neighbor_weights_state[i][key].to(self.device) for i in range(len(neighbor_weights_state)))
            average_weights[key] = weighted_sum / len(neighbor_weights_state)

        self.global_model.load_state_dict(average_weights)

    def train(self):
        """
        - 普通客户端：训练前可调用 aggregate()（由上层决定），训练本体走 _local_train()。
        - 延迟客户端：
            * 预训练阶段：与普通客户端一致（可聚合 + _local_train()）；
            * 预训练结束：切换对齐模式（监督 + 对齐损失）。
        """
        in_warmup = self.is_delayed and (self._warmup_done < self.warmup_bursts)

        # —— 预训练阶段 or 普通客户端：仅监督训练 —— #
        if (not self.is_delayed) or in_warmup:
            self._local_train()
            # 延迟：完成一个 burst 计数
            if in_warmup:
                self._warmup_done += 1

        if self.is_delayed:
            if len(self.dkm_layers) == 0:
                self._register_dkm_layers()

            self._all_teacher_info()
            self._local_train(is_lambda=True)

        self.cluster_model = self._cluster_and_prune_model_weights()

    def set_init_model(self, model: torch.nn.Module):
        self.model = deepcopy(model).to(self.device)
        self.global_model = deepcopy(model)
        self._warmup_done = 0

    def send_model(self):
        """
        返回 (clustered_state_dict, centroids_dict, labels_dict, meta)：
        - 普通/预训练阶段：我们也给出结构化版本（从当前模型做一次聚类），方便对端（对齐端）直接使用；
        - 对齐阶段：直接复用训练后缓存的 cluster_model。
        """
        if self.cluster_model is None:
            clustered_state_dict, cents, labels = self._cluster_and_prune_model_weights()
        else:
            clustered_state_dict, cents, labels = self.cluster_model
        meta = self._prepare_maturity_meta(cents, labels)

        return clustered_state_dict, cents, labels, meta
