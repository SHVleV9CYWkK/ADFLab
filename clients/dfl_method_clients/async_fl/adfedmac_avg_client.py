from __future__ import annotations
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import math

from networkx.classes import neighbors

from clients.client import Client
from utils.kmeans import TorchKMeans


class ADFedMACClient(Client):
    """
    ADFedMAC v3 (W-Space 聚合 + EMD 相似度):
    - 聚合方式: W-Space "普通平均" (已恢复)
    - 相似度 (S): 1D Wasserstein (EMD) 距离 (新), 对 K-Means 不稳定更鲁棒
    - 成熟度 (M): (SWAG 精度 × 稳定性) (保留)
    - 权重 (W): (S × M^γ × R)
    """

    # ------------------------- 初始化 -------------------------
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        hp = dict(hyperparam)
        super().__init__(client_id, dataset_index, full_dataset, hp, device)

        # ... (缓存, 掩码, 历史等保持不变) ...
        self.cluster_model: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None
        self.mask: Dict[str, torch.Tensor] = {}
        self._local_hist: Dict[str, deque] = {}
        self._local_mask_hist: Dict[str, deque] = {}
        self._local_sem: Dict[str, torch.Tensor] = {}

        # —— 必要超参 ——
        self.n_clusters: int = int(hp.get('n_clusters', 16))
        self.epochs: int = int(hp.get('epochs', 1))

        # ... (成熟度, 训练超参保持不变) ...
        self.use_maturity: bool = bool(hp.get("use_maturity", False))
        self.maturity_tau: float = float(hp.get('maturity_tau', 10.0))
        self.maturity_eps: float = float(hp.get('maturity_eps', 1e-5))
        self.beta_drift: float = float(hp.get('beta_drift', 2.0))
        self.beta_invar: float = float(hp.get('beta_invar', 1.0))
        self.beta_mask: float = float(hp.get('beta_mask', 2.0))
        self.history_maxlen: int = int(hp.get('history_maxlen', 20))
        self.gamma_maturity: float = float(hp.get('gamma_maturity', 1.0))
        self.sim_eps: float = 1e-8
        self.apply_mask_every_epoch: bool = bool(hp.get('apply_mask_every_epoch', True))
        self.ps_mass: float = 1.0

        self.ps_mass: float = 1.0  # push-sum 质量，后面会被成熟度初始化覆盖
        self.node_importance: float = 1.0  # 由成熟度得到的节点重要性
        self.ps_mass_initialized: bool = False  # 是否已经用成熟度初始化过 ps_mass


        # ------------------------- 工具：排序/方差 -------------------------
    # ... ( _sort_centroids_and_remap, _cluster_intra_var 保持不变) ...
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

    # ------------------------- 聚类 + 掩码 -------------------------
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
                clustered[key] = w.detach().to(w.device)
                mask[key] = torch.ones_like(w, dtype=torch.bool, device=w.device)

        self.mask = mask
        self._update_local_history(cents, mask)
        return clustered, cents, labels

    def _apply_prune_mask_inplace(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.mask:
                    p.mul_(self.mask[name].to(p.device))

    # ------------------------- 成熟度 (M) -------------------------
    def _c_swag_precision(self, hist: deque) -> torch.Tensor:
        last = hist[-1]
        if len(hist) < 2:
            return last.new_zeros(last.shape[0], 1)
        stack = torch.stack(list(hist), dim=0)
        mu = stack.mean(dim=0)
        var = ((stack - mu) ** 2).mean(dim=0)
        cv = var / (mu.abs() + 1e-4)
        lam = torch.exp(-self.maturity_tau * cv).clamp_min(self.maturity_eps)
        return lam

    def _stability_scores(self, layer_key: str,
                          cents_now: torch.Tensor,
                          labels_now: torch.Tensor,
                          mask_now: torch.Tensor) -> torch.Tensor:
        dev = cents_now.device
        hist = self._local_hist.get(layer_key, [])
        if len(hist) >= 3:
            c_t = hist[-1].to(dev)
            c_t1 = hist[-2].to(dev)
            c_t2 = hist[-3].to(dev)
            accel = torch.abs((c_t - c_t1) - (c_t1 - c_t2))  # 加速度
        else:
            accel = torch.zeros_like(cents_now)
        flat_w = self.model.state_dict()[layer_key].to(dev).view(-1, 1).detach()
        K = cents_now.shape[0]
        invar = self._cluster_intra_var(flat_w, labels_now, K, eps=self.maturity_eps)
        if len(self._local_mask_hist.get(layer_key, [])) >= 2:
            prev_m = self._local_mask_hist[layer_key][-2].to(dev)
            flips = (prev_m ^ mask_now).float().mean()
        else:
            flips = torch.tensor(0.0, device=dev)
        stab = torch.exp(- self.beta_drift * accel - self.beta_invar * invar - self.beta_mask * flips)
        return stab

    def _local_maturity(self, layer_key: str,
                        cents_now: torch.Tensor,
                        labels_now: torch.Tensor,
                        mask_now: torch.Tensor) -> torch.Tensor:
        if layer_key not in self._local_hist:
            return torch.zeros_like(cents_now)
        lam = self._c_swag_precision(self._local_hist[layer_key]).to(cents_now.device)
        stab = self._stability_scores(layer_key, cents_now, labels_now, mask_now)
        return lam * stab

    def _compute_node_maturity(self,
                               cents: Dict[str, torch.Tensor],
                               labels: Dict[str, torch.Tensor]) -> float:
        layer_ms = []
        layer_sizes = []

        # 确保 mask 有东西
        if not self.mask:
            self.mask = {k: torch.ones_like(v, dtype=torch.bool, device=v.device)
                         for k, v in self.model.state_dict().items()}

        for layer_key, c_now in cents.items():
            if c_now is None:
                continue
            self._ensure_hist_slot(layer_key)

            m_now = self.mask.get(layer_key, None)
            l_now = labels.get(layer_key, None)
            if m_now is None or l_now is None:
                continue

            c_now = c_now.to(self.device)
            m_now = m_now.to(self.device)
            l_now = l_now.to(self.device)

            maturity_vec = self._local_maturity(layer_key, c_now, l_now, m_now)
            # 这一层的平均成熟度
            m_layer = float(maturity_vec.mean().item())
            layer_ms.append(m_layer)
            layer_sizes.append(self._get_layer_size(layer_key))

        if not layer_ms:
            # 没有历史/信息时，退化为 1.0
            return 1.0

        ms_t = torch.tensor(layer_ms, device=self.device).float()
        sz_t = torch.tensor(layer_sizes, device=self.device).float()
        weights = sz_t / (sz_t.sum() + 1e-8)
        M = float((ms_t * weights).sum().item())
        return M

    def _prepare_maturity_meta(self, cents: Dict[str, torch.Tensor],
                               labels: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # 1) 先基于当前 cluster 结果算“节点成熟度” M_i
        node_maturity = self._compute_node_maturity(cents, labels)
        # 防止数值问题，做个裁剪
        node_maturity = max(self.maturity_eps, float(node_maturity))

        # 2) 把成熟度映射到节点重要性 c_i
        importance = (node_maturity ** self.gamma_maturity)

        # 3) 用成熟度初始化 push-sum 质量（只做一次，后续轮次不再重复）
        if not self.ps_mass_initialized:
            self.node_importance = importance
            self.ps_mass = float(self.node_importance)
            self.ps_mass_initialized = True

        # 4) 正常做 push-sum 的等分（这一段是你原来的逻辑）
        d_out = self.k_push + 1  # +1 表示“留给自己”的那一份
        share = float(self.ps_mass) / float(d_out)
        # 重置本地质量为自己保留的一份
        self.ps_mass = share

        return {
            'version_meta': 'adfedmac_meta_v3',
            'sender_id': self.id,
            'version': self.local_version,
            'sender_time': self.last_update_time,
            "ps_mass_share": share,
        }

    def _get_layer_size(self, layer_key: str) -> int:
        m = self.mask.get(layer_key, None)
        if m is not None:
            return int(m.to(torch.int).sum().item())
        if (self.cluster_model is not None) and (self.cluster_model[2].get(layer_key) is not None):
            return int(self.cluster_model[2][layer_key].numel())
        return int(self.model.state_dict()[layer_key].numel())

    # ------------------------- 训练 -------------------------
    def _local_train(self):
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

    # ------------------------- 聚合 (W-Space Averaging) -------------------------
    @torch.no_grad()
    def aggregate(self):
        if len(self.neighbor_model_weights_buffer) == 0:
            return  # 无邻居或无训练就不聚合

        # 启动逻辑
        if self.cluster_model is None:
            neighbor_states = [tpl[0] for tpl in self.neighbor_model_weights_buffer]
            if len(neighbor_states) == 0:
                return
            avg_state = {}
            keys = set(neighbor_states[0].keys())
            for k in keys:
                acc = torch.stack([st[k].to(self.device) for st in neighbor_states]).mean(dim=0)
                avg_state[k] = acc
            self.model.load_state_dict(avg_state)
            self.cluster_model = self._cluster_and_prune_model_weights()

            recv_mass = sum(float(tpl[-1]['ps_mass_share']) for tpl in self.neighbor_model_weights_buffer
                            if isinstance(tpl, tuple) and len(tpl) >= 2)
            self.ps_mass = float(self.ps_mass) + float(recv_mass)
            self.neighbor_model_weights_buffer.clear()
            return

        neighbor_states = []
        neighbor_masses = []
        for tpl in self.neighbor_model_weights_buffer:
            neighbor_states.append(tpl[0])
            if len(tpl) >= 2:
                neighbor_masses.append(float(tpl[-1]['ps_mass_share']))
            else:
                neighbor_masses.append(0.0)

        keys = set(neighbor_states[0].keys())
        local_mass = float(self.ps_mass)

        total_mass = local_mass + float(sum(neighbor_masses))
        if total_mass <= 0:
            # 极端兜底：退化成均匀平均
            total_mass = 1.0
            local_mass = 1.0 / (len(neighbor_states) + 1.0)
            neighbor_masses = [1.0 / (len(neighbor_states) + 1.0)] * len(neighbor_states)

        avg = {}
        local_state_dict = self.model.state_dict()
        for k in keys:
            acc = None
            # 邻居累加：每个张量乘以该邻居的 ps_mass 再加和
            for st, m in zip(neighbor_states, neighbor_masses):
                if k not in st:
                    continue
                t = st[k].to(self.device)
                acc = t.mul(m) if acc is None else acc.add(t, alpha=m)
            # 再加本地（带本地质量）
            if k in local_state_dict:
                t_local = local_state_dict[k].to(self.device)
                if acc is None:
                    acc = t_local.mul(local_mass)
                else:
                    acc = acc.add(t_local, alpha=local_mass)
            # 最后除以总质量
            avg[k] = acc.div(total_mass)
        self.ps_mass = float(total_mass)

        self.model.load_state_dict(avg)
        self.neighbor_model_weights_buffer.clear()

    # ------------------------- 训练总流程 -------------------------
    def train(self):
        self._local_train()
        self.cluster_model = self._cluster_and_prune_model_weights()

    # ------------------------- 初始化 / 发送 -------------------------
    def set_init_model(self, model: nn.Module):
        self.model = deepcopy(model).to(self.device)

    def send_model(self):
        if self.cluster_model is None:
            clustered_state_dict, cents, labels = self._cluster_and_prune_model_weights()
            self.cluster_model = (clustered_state_dict, cents, labels)
        else:
            clustered_state_dict, cents, labels = self.cluster_model

        # 打包成熟度 meta
        meta = self._prepare_maturity_meta(cents, labels)

        return clustered_state_dict, cents, labels, meta