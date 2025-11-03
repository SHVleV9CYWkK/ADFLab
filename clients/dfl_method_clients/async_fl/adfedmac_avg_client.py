from __future__ import annotations
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import math

from clients.client import Client
from utils.kmeans import TorchKMeans


class ADFedMACClient(Client):
    """
    ADFedMAC: 仅依赖“质心统计”的去中心化异步聚合客户端
    - 聚类 -> 质心/标签/掩码
    - 相似度（数值 × 语义，可退化为纯数值）
    - 成熟度（历史稳定性 × SWAG 精度）
    - 新鲜度（时延衰减）
    - 加权平均聚合
    """

    # ------------------------- 初始化 -------------------------
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        hp = dict(hyperparam)
        super().__init__(client_id, dataset_index, full_dataset, hp, device)

        # —— 结构缓存 ——
        # 邻居缓存：四元组 (clustered_state, centroids, labels, meta)
        self.neighbor_model_weights_buffer: List[Any] = []

        # 本地聚类缓存（避免重复聚类）
        self.cluster_model: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None

        # 掩码与历史（用于成熟度）
        self.mask: Dict[str, torch.Tensor] = {}
        self._local_hist: Dict[str, deque] = {}       # 每层质心历史，deque[[K,1]]，长度限制
        self._local_mask_hist: Dict[str, deque] = {}  # 每层掩码历史，deque[bool tensor]

        # （可选）本地语义签名表：{ layer_key: Tensor[K,C] }；若你实现了，就放进来
        self._local_sem: Dict[str, torch.Tensor] = {}

        # —— 必要超参 ——
        self.n_clusters: int = int(hp.get('n_clusters', 16))
        self.epochs: int = int(hp.get('epochs', 1))

        # 相似度融合
        self.alpha_sem: float = float(hp.get('alpha_sem', 0.5))        # 语义占比 α ∈ [0,1]
        self.beta_num: float = float(hp.get('beta_num', 2.0))          # 数值温度

        # 成熟度 & 历史
        self.maturity_tau: float = float(hp.get('maturity_tau', 1e4)) # SWAG 精度的严格度
        self.maturity_eps: float = float(hp.get('maturity_eps', 1e-5))
        self.beta_drift: float = float(hp.get('beta_drift', 2.0))
        self.beta_invar: float = float(hp.get('beta_invar', 1.0))
        self.beta_mask: float = float(hp.get('beta_mask', 2.0))
        self.history_maxlen: int = int(hp.get('history_maxlen', 20))

        # 新鲜度与聚合权重
        self.lambda_time: float = float(hp.get('lambda_time', 0.05))
        self.gamma_maturity: float = float(hp.get('gamma_maturity', 1.0))
        self.sim_eps: float = 1e-8

        # 训练阶段
        self.apply_mask_every_epoch: bool = bool(hp.get('apply_mask_every_epoch', True))

    # ------------------------- 工具：质心排序/重映射/方差 -------------------------
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

    # ------------------------- 历史缓存（成熟度） -------------------------
    def _ensure_hist_slot(self, layer: str):
        if layer not in self._local_hist:
            self._local_hist[layer] = deque(maxlen=self.history_maxlen)
        if layer not in self._local_mask_hist:
            self._local_mask_hist[layer] = deque(maxlen=self.history_maxlen)

    def _update_local_history(self, centroids: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]):
        for layer, c in centroids.items():
            self._ensure_hist_slot(layer)
            self._local_hist[layer].append(c.detach())
        for layer, m in mask.items():
            self._ensure_hist_slot(layer)
            self._local_mask_hist[layer].append(m.detach())

    # ------------------------- 聚类 + 掩码（不写回模型） -------------------------
    def _cluster_and_prune_model_weights(self):
        """
        返回:
          clustered: 聚类重构后的权重（未写回）
          cents:     每层质心 [K,1]
          labels:    每层标签 [N]
        同时更新 self.mask 并记录历史
        """
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
                clustered[key] = w.detach().to(w.device)
                mask[key] = torch.ones_like(w, dtype=torch.bool, device=w.device)

        self.mask = mask
        self._update_local_history(cents, mask)
        return clustered, cents, labels

    # —— 就地应用掩码 ——
    def _apply_prune_mask_inplace(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.mask:
                    p.mul_(self.mask[name].to(p.device))

    # ------------------------- 成熟度估计 -------------------------
    def _c_swag_precision(self, hist: deque) -> torch.Tensor:
        # SWAG 风格相对方差 → 精度（大表示稳定）
        last = hist[-1]
        if len(hist) < 2:
            return last.new_zeros(last.shape[0], 1)

        stack = torch.stack(list(hist), dim=0)  # [T,K,1]
        mu = stack.mean(dim=0)                  # [K,1]
        var = ((stack - mu) ** 2).mean(dim=0)   # [K,1]

        cv = var / (mu.abs() + 1e-4)            # 相对方差
        lam = torch.exp(-self.maturity_tau * cv).clamp_min(self.maturity_eps)
        return lam

    def _stability_scores(self, layer_key: str,
                          cents_now: torch.Tensor,
                          labels_now: torch.Tensor,
                          mask_now: torch.Tensor) -> torch.Tensor:
        dev = cents_now.device
        # 漂移（与上一帧质心差）
        if len(self._local_hist.get(layer_key, [])) >= 2:
            prev_c = self._local_hist[layer_key][-2].to(dev)
            drift = torch.abs(cents_now - prev_c)  # [K,1]
        else:
            drift = torch.zeros_like(cents_now)

        # 簇内方差（当前帧）
        flat_w = self.model.state_dict()[layer_key].to(dev).view(-1, 1).detach()
        K = cents_now.shape[0]
        invar = self._cluster_intra_var(flat_w, labels_now, K, eps=self.maturity_eps)  # [K,1]

        # 掩码翻转率（层级标量）
        if len(self._local_mask_hist.get(layer_key, [])) >= 2:
            prev_m = self._local_mask_hist[layer_key][-2].to(dev)
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
        return lam * stab  # [K,1]

    def _prepare_maturity_meta(self, cents: Dict[str, torch.Tensor],
                               labels: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        layer_maturity = {}
        # for layer_key, c_now in cents.items():
        #     if c_now is None:
        #         continue
        #     self._ensure_hist_slot(layer_key)
        #     m_now = self.mask[layer_key].to(c_now.device)
        #     l_now = labels[layer_key]
        #     maturity_vec = self._local_maturity(layer_key, c_now, l_now, m_now)
        #     layer_maturity[layer_key] = float(maturity_vec.mean().item())

        return {
            'version_meta': 'adfedmac_meta_v1',
            'sender_id': self.id,
            'version': int(getattr(self, "local_version", 0)),
            'sender_time': float(getattr(self, "last_update_time", 0.0)),
            'layer_maturity': layer_maturity
        }

    def _soft_jaccard_from_centroids(self,
                                     C: torch.Tensor,  # [K,D]
                                     C_peer: torch.Tensor,  # [K,D]
                                     a: torch.Tensor,  # [K] 本地簇占比
                                     b: torch.Tensor,  # [K] 邻居簇占比
                                     beta_sem: float = 5.0,
                                     eps: float = 1e-8) -> torch.Tensor:
        """
        用“簇→对方簇”的 softmax 构造 soft-Jaccard：
          P_{i→j} = softmax_j(-β_sem * ||C_i - C'_j||^2 / med)
          Q_{j→i} = softmax_i(-β_sem * ||C'_j - C_i||^2 / med)
          inter_ij = sqrt( a_i * P_{i→j} * b_j * Q_{j→i} )
          union_ij = a_i + b_j - inter_ij
          J_ij = inter_ij / (union_ij + eps)
        返回 J:[K,K] ∈ (0,1]
        """
        K = C.shape[0]
        D2 = torch.cdist(C, C_peer, p=2).pow(2)  # [K,K]
        scale = D2.median() + eps

        logits_L2R = - D2 / scale * beta_sem  # [K,K]
        logits_R2L = logits_L2R.t().contiguous()  # [K,K]

        P = torch.softmax(logits_L2R, dim=1)  # local→peer
        Q = torch.softmax(logits_R2L, dim=1)  # peer →local

        a = a.view(K, 1)  # [K,1]
        b = b.view(1, K)  # [1,K]
        inter = torch.sqrt((a * P) * (b * Q.t()))  # [K,K]
        union = a + b - inter
        J = inter / (union + eps)
        return J.clamp(0.0, 1.0)

    def _layer_similarity_mix(self,
                                  C: torch.Tensor, C_peer: torch.Tensor,  # [K,D]
                                  a: torch.Tensor, b: torch.Tensor,  # [K], [K]
                                  alpha_sem: float = 0.5,
                                  beta_num: float = 2.0,
                                  beta_sem: float = 5.0,
                                  eps: float = 1e-8) -> torch.Tensor:
        """
        DKM 样式融合：
          S: 数值相似（exp(-β_num * d^2 / med)）      → 置信度
          J: 语义相似（上面的 soft-Jaccard）          → 语义一致性
          M = (J+eps)^{α} * (S+eps)^{1-α}
          A = softmax_j( log M )                    → 软匹配
          s_layer = mean_i Σ_j A_ij * S_ij          → 层标量相似度 (0,1]
        """
        # 数值相似
        D2 = torch.cdist(C, C_peer, p=2).pow(2)  # [K,K]
        scale = D2.median() + eps
        S = torch.exp(- beta_num * D2 / scale)  # [K,K] in (0,1]

        # soft-Jaccard（语义）
        J = self._soft_jaccard_from_centroids(C, C_peer, a, b, beta_sem=beta_sem, eps=eps)  # [K,K]

        # 融合 & 行 softmax
        logits = (1 - alpha_sem) * (S + eps).log() + alpha_sem * (J + eps).log()
        A = torch.softmax(logits, dim=1)  # [K,K]

        s_layer = (A * S).sum(dim=1).mean()  # scalar
        return (1.0 - torch.exp(-s_layer)).clamp(0.0, 1.0)

    def _neighbor_similarity(self,
                                 local_cents: Dict[str, torch.Tensor],
                                 neighbor_cents: Dict[str, torch.Tensor],
                                 local_labels: Dict[str, torch.Tensor] | None,
                                 neighbor_labels: Dict[str, torch.Tensor] | None,
                                 alpha_sem: float,
                                 beta_num: float,
                                 beta_sem: float = 5.0) -> float:
        """
        对每层：
          - C, C_peer → [K,D]
          - a, b 来自 labels 的计数占比；若无 labels 则 a=b=均匀
          - 得到 s_layer（上面函数）
        最后按层规模（被保留参数量）做加权平均，得出该邻居的整体相似度 s ∈ (0,1]。
        """
        per_s, per_sz = [], []
        sd = self.model.state_dict()

        for layer_key, C in local_cents.items():
            C_peer = neighbor_cents.get(layer_key, None)
            if C is None or C_peer is None:
                continue

            C = C.view(C.shape[0], -1).float().to(self.device)
            C_peer = C_peer.view(C_peer.shape[0], -1).float().to(self.device)
            K = C.shape[0]

            # 簇占比 a（本地），b（邻居）
            if (local_labels is not None) and (local_labels.get(layer_key) is not None):
                la = local_labels[layer_key].to(self.device).view(-1)
                a_counts = torch.bincount(la, minlength=K).float()
            else:
                a_counts = torch.ones(K, device=self.device)
            if (neighbor_labels is not None) and (neighbor_labels.get(layer_key) is not None):
                lb = neighbor_labels[layer_key].to(self.device).view(-1)
                b_counts = torch.bincount(lb, minlength=K).float()
            else:
                b_counts = torch.ones(K, device=self.device)
            a = a_counts / (a_counts.sum() + 1e-8)
            b = b_counts / (b_counts.sum() + 1e-8)

            sL = self._layer_similarity_mix(
                C, C_peer, a, b,
                alpha_sem=alpha_sem, beta_num=beta_num, beta_sem=beta_sem
            )
            per_s.append(sL)

            # 层规模：用掩码保留数或参数总数
            m = self.mask.get(layer_key, None)
            if m is not None:
                per_sz.append(int(m.to(torch.int).sum().item()))
            else:
                per_sz.append(int(sd[layer_key].numel()))

        if len(per_s) == 0:
            return 0.5

        s = torch.stack(per_s).to(self.device)
        w = torch.tensor(per_sz, device=self.device).float()
        w = w / (w.sum() + 1e-8)
        return float((s * w).sum().item())

    # ------------------------- 邻居权重（相似度 × 成熟度^γ × 新鲜度） -------------------------
    @torch.no_grad()
    def _compute_peer_weights(self,
                              local_cents: Dict[str, torch.Tensor]) -> List[float]:
        if len(self.neighbor_model_weights_buffer) == 0:
            return []

        t_now = float(getattr(self, "last_update_time", 0.0))
        scores = []

        for (peer_state, peer_cents, peer_labels, peer_meta) in self.neighbor_model_weights_buffer:
            # 相似度：数值×语义
            s = self._neighbor_similarity(
                local_cents=local_cents,
                neighbor_cents=peer_cents,
                local_labels=self.cluster_model[2] if (self.cluster_model is not None) else None,
                neighbor_labels=peer_labels,
                alpha_sem=self.alpha_sem,
                beta_num=self.beta_num,
                beta_sem=getattr(self, "beta_sem", 5.0),
            )

            # 成熟度：按层规模加权均值
            # m = 1.0
            # if isinstance(peer_meta, dict) and ('layer_maturity' in peer_meta):
            #     ms, szs = [], []
            #     for layer_key, c in local_cents.items():
            #         if c is None:
            #             continue
            #         ml = peer_meta['layer_maturity'].get(layer_key, None)
            #         if ml is None:
            #             continue
            #         ms.append(float(ml))
            #         szs.append(self._layer_size(layer_key, None))
            #     if ms:
            #         ms_t = torch.tensor(ms, device=self.device).float()
            #         sz_t = torch.tensor(szs, device=self.device).float()
            #         m = float((ms_t * (sz_t / (sz_t.sum() + 1e-8))).sum().item())

            # 新鲜度
            sender_time = float(peer_meta.get('sender_time', t_now)) if isinstance(peer_meta, dict) else t_now
            r = math.exp(- self.lambda_time * max(0.0, t_now - sender_time))

            score = max(self.sim_eps, s) * r
            scores.append(score)

        Z = sum(scores) + self.sim_eps
        return [s / Z for s in scores]

    # ------------------------- 训练 -------------------------
    def _local_train(self):
        """
        标准监督训练；就地应用掩码（避免每个 batch load_state_dict）。
        建议：在每个 epoch 开始时应用一次掩码；若想更严格，可每 N step 应用一次。
        """
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader not initialized; ensure init_client() on join.")

        self.model.train()
        for _ in range(self.epochs):
            if self.apply_mask_every_epoch and (len(self.mask) > 0):
                self._apply_prune_mask_inplace()

            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()

    # ------------------------- 聚合（加权均值 + 自锚点） -------------------------
    @torch.no_grad()
    def aggregate(self):
        """
        动态自锚点：若有 n 个邻居，则本地模型权重 = 1/(n+1)，
        邻居权重占剩余的 1 - 1/(n+1)，并按相似度×成熟度×新鲜度在邻居间分配。
        """
        n = len(self.neighbor_model_weights_buffer)
        if n == 0:
            return  # 无邻居或无训练就不聚合

        if self.cluster_model is None:
            # 通过邻居的质心聚合出初始模型
            neighbor_states = [tpl[0] for tpl in self.neighbor_model_weights_buffer]
            if len(neighbor_states) == 0:
                return

            # 初始化为邻居模型的简单平均
            avg_state = {}
            keys = set(neighbor_states[0].keys())

            for k in keys:
                acc = None
                for st in neighbor_states:
                    t = st[k].to(self.device)
                    acc = t if acc is None else acc.add(t)
                avg_state[k] = acc.div(len(neighbor_states))

            # 写回到模型并缓存 cluster_model
            self.model.load_state_dict(avg_state)
            # 立即做一次聚类，保证后续能计算相似度
            self.cluster_model = self._cluster_and_prune_model_weights()


        # 1) 本地最新质心（用于相似度）
        _, local_cents, _ = self.cluster_model

        # 2) 邻居相对权重（和为 1）
        neighbor_rel = self._compute_peer_weights(local_cents)  # 长度 n，∑=1
        if len(neighbor_rel) != n:
            return

        neighbor_states = [tpl[0] for tpl in self.neighbor_model_weights_buffer]
        keys = set(neighbor_states[0].keys())

        # 3) 动态自锚点
        local_w = 1.0 / (n + 1.0)  # 本地 1/(n+1)
        scale_neighbors = 1.0 - local_w  # 邻居总权重
        neighbor_w = [w * scale_neighbors for w in neighbor_rel]  # 仍然 ∑=1-local_w

        # 4) 计算加权均值（邻居 + 本地）
        avg = {}
        # 先邻居
        for k in keys:
            acc = None
            for st, w in zip(neighbor_states, neighbor_w):
                t = st[k].to(self.device)
                acc = t.mul(w) if acc is None else acc.add(t, alpha=w)
            # 再加本地锚点
            t_local = self.model.state_dict()[k].to(self.device)
            acc = acc.add(t_local, alpha=local_w)
            avg[k] = acc

        self.model.load_state_dict(avg)


    # ------------------------- 训练总流程 -------------------------
    def train(self):
        # 先做一次本地训练
        self._local_train()
        # 更新本地聚类缓存
        self.cluster_model = self._cluster_and_prune_model_weights()

    # ------------------------- 初始化 / 发送 -------------------------
    def set_init_model(self, model: nn.Module):
        self.model = deepcopy(model).to(self.device)

    def send_model(self):
        """
        返回四元组：(clustered_state_dict, cents, labels, meta)
        meta 包含：
          - layer_maturity: {layer: float}
          - layer_semantics: {layer: Tensor[K,C]}（可选；若为空则在对方退化为纯数值相似）
          - sender_id/version/sender_time
        """
        if self.cluster_model is None:
            clustered_state_dict, cents, labels = self._cluster_and_prune_model_weights()
        else:
            clustered_state_dict, cents, labels = self.cluster_model

        meta = self._prepare_maturity_meta(cents, labels)

        return clustered_state_dict, cents, labels, meta