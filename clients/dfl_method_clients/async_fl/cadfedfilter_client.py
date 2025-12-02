from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn

from clients.client import Client
from utils.kmeans import TorchKMeans

# Centroid-Guided Client Filtering for Asynchronous Decentralized Federated Learning (C-ADFedFilter)
class CADFedFilterClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        hp = dict(hyperparam)
        super().__init__(client_id, dataset_index, full_dataset, hp, device)

        self.cluster_model: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None
        self.mask: Dict[str, torch.Tensor] = {}

        self.n_clusters: int = int(hp.get('n_clusters', 16))
        self.epochs: int = int(hp.get('epochs', 1))

        self.gamma_maturity: float = float(hp.get('gamma_maturity', 1.0))
        self.sim_eps: float = 1e-8
        self.apply_mask_every_epoch: bool = bool(hp.get('apply_mask_every_epoch', True))

        self.ps_mass: float = 1.0

        self.use_global_cents: bool = bool(hp.get('use_global_cents', True))
        self.global_cents: Dict[str, torch.Tensor] = {}

        # ============================================================
        #  SAF：Semantic-Aware Filtering（核心参数）
        # ============================================================
        self.use_saf: bool = bool(hp.get("use_saf", True))
        # drift > tau 就过滤这个邻居的模型（mass 不过滤）
        self.saf_tau: float = float(hp.get("saf_tau", 5e-4))
        # 是否把 drift 打印出来调试
        self.print_drift: bool = bool(hp.get("print_drift", False))

    # ============================================================
    # KMeans 聚类辅助
    # ============================================================
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
        clustered = {}
        mask = {}
        cents = {}
        labels = {}

        state = self.model.state_dict()
        for key, w in state.items():
            if "weight" in key and "bn" not in key and "downsample" not in key:
                orig = w.shape
                flat = w.detach().view(-1, 1)

                kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)

                if self.use_global_cents:
                    g = self.global_cents.get(key, None)
                    if g is not None and g.shape[0] == self.n_clusters:
                        kmeans.init_centroids = g.to(flat.device)

                optimized_fit = torch.compile(kmeans.fit)
                optimized_fit(flat)

                cent_sorted, lab_sorted = self._sort_centroids_and_remap(
                    kmeans.centroids.view(-1, 1), kmeans.labels_
                )

                cent_sorted = cent_sorted.to(self.device)

                new_w = cent_sorted[lab_sorted].view(orig)
                is_zero = cent_sorted.view(-1) == 0
                m = (is_zero[lab_sorted].view(orig) == 0)

                clustered[key] = new_w
                mask[key] = m.bool()
                cents[key] = cent_sorted
                labels[key] = lab_sorted.view(-1).to(self.device)
            else:
                clustered[key] = w.detach().to(self.device)
                mask[key] = torch.ones_like(w, dtype=torch.bool, device=w.device)

        self.mask = mask

        if self.use_global_cents:
            for k, c in cents.items():
                if (k not in self.global_cents) or (self.global_cents[k].shape != c.shape):
                    self.global_cents[k] = c.detach().clone()

        return clustered, cents, labels

    # ============================================================
    # SAF：语义漂移计算
    # ============================================================
    def _compute_neighbor_centroid_drift(self, nb_cents: Dict[str, torch.Tensor]) -> float:
        """
        drift = mean over layers of L2^2 distance between neighbor cents & global cents
        """
        if (
            not self.use_saf
            or not self.use_global_cents
            or len(self.global_cents) == 0
            or nb_cents is None
            or len(nb_cents) == 0
        ):
            return 0.0

        diffs = []

        for key, g_c in self.global_cents.items():
            if key not in nb_cents:
                continue
            c_nb = nb_cents[key].to(self.device)
            g_c = g_c.to(self.device)

            if c_nb.shape != g_c.shape:
                continue

            diff = (c_nb - g_c).view(-1)
            diffs.append(diff.pow(2).mean())

        if not diffs:
            return 0.0

        return float(torch.stack(diffs).mean().item())

    # ============================================================
    # 本地训练
    # ============================================================
    def _apply_prune_mask_inplace(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.mask:
                    p.mul_(self.mask[name].to(self.device))

    def _local_train(self):
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader not initialized")

        self.model.train()
        for _ in range(self.epochs):
            if self.apply_mask_every_epoch and len(self.mask) > 0:
                self._apply_prune_mask_inplace()

            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()

    # ============================================================
    # 重构邻居模型（解码）
    # ============================================================
    def _reconstruct_from_compressed(self, cents, labels, uncompressed):
        reconstructed = {}

        for k, v in uncompressed.items():
            reconstructed[k] = v.to(self.device)

        local_state = self.model.state_dict()

        for key, idx in labels.items():
            if key not in cents:
                continue

            c = cents[key].to(self.device)
            idx = idx.to(self.device).long()

            if key not in local_state:
                continue

            orig_shape = local_state[key].shape
            w_recon = c[idx].view(orig_shape)
            reconstructed[key] = w_recon

        return reconstructed

    def _prepare_maturity_meta(self) -> Dict[str, Any]:
        d_out = self.k_push + 1
        share = float(self.ps_mass) / float(d_out)
        self.ps_mass = share

        return {
            'version_meta': 'cadfedfilter_meta_v1',
            'sender_id': self.id,
            'version': self.local_version,
            'sender_time': self.last_update_time,
            "ps_mass_share": share,
        }

    # ============================================================
    # 聚合（指数衰减软权重版）
    # ============================================================
    @torch.no_grad()
    def aggregate(self):

        if len(self.neighbor_model_weights_buffer) == 0:
            return

        # ------------------------ 首次聚合逻辑 ------------------------
        if self.cluster_model is None:
            neighbor_states = [tpl[0] for tpl in self.neighbor_model_weights_buffer]
            keys = set(neighbor_states[0].keys())

            avg_state = {}
            for k in keys:
                avg_state[k] = torch.stack(
                    [st[k].to(self.device) for st in neighbor_states]
                ).mean(dim=0)

            self.model.load_state_dict(avg_state)
            self.cluster_model = self._cluster_and_prune_model_weights()

            # ps_mass 初次累加所有邻居 Quality Mass
            recv_mass = sum(
                float(tpl[-1]['ps_mass_share'])
                for tpl in self.neighbor_model_weights_buffer
                if isinstance(tpl[-1], dict) and 'ps_mass_share' in tpl[-1]
            )

            self.ps_mass += float(recv_mass)
            self.neighbor_model_weights_buffer.clear()
            return

        # ------------------------ 常规聚合 ------------------------
        neighbor_states = []
        neighbor_masses = []
        neighbor_cents_list = []

        for tpl in self.neighbor_model_weights_buffer:
            state_dict = tpl[0]
            meta = tpl[-1] if isinstance(tpl[-1], dict) else {}

            mass_share = float(meta.get("ps_mass_share", 0.0))

            neighbor_states.append(state_dict)
            neighbor_masses.append(mass_share)

            if len(tpl) >= 2 and isinstance(tpl[1], dict):
                neighbor_cents_list.append(tpl[1])
            else:
                neighbor_cents_list.append({})

        keys = set(neighbor_states[0].keys())
        local_mass = float(self.ps_mass)

        # ==========================================================
        # SAF-soft：为每个邻居计算 alpha_j = exp(-gamma * drift_j)
        # 并对模型 & mass 一致加权（不再做 hard drop）
        # ==========================================================
        eff_masses = []      # 有效质量 m_eff = alpha_j * m_raw
        eff_cents = []       # 对应的邻居质心
        alphas = []          # 仅用于调试

        for m_raw, c_nb in zip(neighbor_masses, neighbor_cents_list):

            if (
                self.use_saf
                and self.use_global_cents
                and len(self.global_cents) > 0
                and c_nb is not None
                and len(c_nb) > 0
            ):
                drift_j = self._compute_neighbor_centroid_drift(c_nb)

                if self.print_drift:
                    print(f"[Client {self.id}] neighbor drift = {drift_j:.6f}")

                # 指数衰减：alpha_j = exp(-gamma * drift_j)
                alpha_j = math.exp(-self.saf_gamma * drift_j)
                # 限制在 [saf_min_alpha, 1.0]
                alpha_j = max(self.saf_min_alpha, min(1.0, alpha_j))
            else:
                alpha_j = 1.0

            alphas.append(alpha_j)
            eff_masses.append(alpha_j * m_raw)
            eff_cents.append(c_nb)

        # ------------------------ Push-sum 质量更新 ------------------------
        total_mass = local_mass + sum(eff_masses)
        if total_mass <= 0:
            # 极端兜底：退化成仅本地
            total_mass = local_mass if local_mass > 0 else 1.0
            eff_masses = [0.0 for _ in eff_masses]

        self.ps_mass = float(total_mass)

        # ------------------------ 模型聚合（使用软权重后的有效质量） ------------------------
        avg = {}
        local_state = self.model.state_dict()

        for k in keys:
            acc = None

            # 邻居贡献：每个邻居质量 m_j → m_eff = alpha_j * m_j
            for st, m_eff in zip(neighbor_states, eff_masses):
                if m_eff <= 0.0:
                    continue
                if k not in st:
                    continue
                t = st[k].to(self.device)
                acc = t.mul(m_eff) if acc is None else acc.add(t, alpha=m_eff)

            # 本地贡献
            if local_mass > 0:
                t_local = local_state[k].to(self.device)
                acc = t_local.mul(local_mass) if acc is None else acc.add(t_local, alpha=local_mass)

            avg[k] = acc.div(total_mass)

        self.model.load_state_dict(avg)

        # ------------------------ 更新 global_cents（也用 m_eff） ------------------------
        _, local_cents, _ = self.cluster_model
        new_global_cents = {}

        for key, c_local in local_cents.items():
            c_local = c_local.to(self.device)

            num = c_local * local_mass
            den = local_mass

            for c_nb, m_eff in zip(eff_cents, eff_masses):
                if m_eff <= 0.0:
                    continue
                if key not in c_nb:
                    continue
                num = num + m_eff * c_nb[key].to(self.device)
                den = den + m_eff

            new_global_cents[key] = num / (den + self.sim_eps)

        self.global_cents = new_global_cents

        self.neighbor_model_weights_buffer.clear()

    # ============================================================
    # 训练接口
    # ============================================================
    def train(self):
        self._local_train()
        self.cluster_model = self._cluster_and_prune_model_weights()

    # ============================================================
    # 模型发送 & 接收
    # ============================================================
    def set_init_model(self, model: nn.Module):
        model = deepcopy(model).to(self.device)
        model = torch.compile(model)
        self.model = model

    def send_model(self):
        if self.cluster_model is None:
            clustered_state_dict, cents, labels = self._cluster_and_prune_model_weights()
            self.cluster_model = (clustered_state_dict, cents, labels)
        else:
            clustered_state_dict, cents, labels = self.cluster_model

        uncompressed = {
            k: v for k, v in clustered_state_dict.items() if k not in cents
        }

        payload = {
            'cents': cents,
            'labels': labels,
            'uncompressed': uncompressed,
        }
        meta = self._prepare_maturity_meta()
        return payload, meta

    def receive_neighbor_model(self, neighbor_payload):
        if isinstance(neighbor_payload, tuple) and len(neighbor_payload) >= 2:
            compressed = neighbor_payload[0]
            meta = neighbor_payload[-1]

            if isinstance(compressed, dict) and 'cents' in compressed:
                nb_cents = compressed['cents']
                nb_labels = compressed['labels']
                nb_uncompressed = compressed['uncompressed']

                reconstructed = self._reconstruct_from_compressed(
                    nb_cents, nb_labels, nb_uncompressed
                )
                new_payload = (reconstructed, nb_cents, nb_labels, meta)
            else:
                new_payload = neighbor_payload
        else:
            new_payload = neighbor_payload

        super().receive_neighbor_model(new_payload)
