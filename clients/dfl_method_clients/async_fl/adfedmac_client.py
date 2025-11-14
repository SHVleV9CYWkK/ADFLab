from __future__ import annotations
from collections import deque
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from clients.client import Client
from utils.kmeans import TorchKMeans


class ADFedMACClient(Client):
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

        self.gamma_maturity: float = float(hp.get('gamma_maturity', 1.0))
        self.sim_eps: float = 1e-8
        self.apply_mask_every_epoch: bool = bool(hp.get('apply_mask_every_epoch', True))
        self.ps_mass: float = 1.0

    @staticmethod
    def _sort_centroids_and_remap(c1d: torch.Tensor, lbl: torch.Tensor):
        c = c1d.view(-1).detach()
        order = torch.argsort(c)
        sorted_c = c[order].view(-1, 1)
        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        new_labels = old2new[lbl]
        return sorted_c, new_labels

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
        return clustered, cents, labels

    def _apply_prune_mask_inplace(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.mask:
                    p.mul_(self.mask[name].to(p.device))

    def _prepare_maturity_meta(self, cents: Dict[str, torch.Tensor],
                               labels: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        d_out = self.k_push + 1
        share = float(self.ps_mass) / float(d_out)
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

    # ------------------------- 聚合 -------------------------
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