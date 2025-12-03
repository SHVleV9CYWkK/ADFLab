from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

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

        self.use_similarity_weight: bool = bool(hp.get("use_similarity_weight", True))
        self.alpha_mix: float = float(hp.get("alpha_mix", 0.6))   # 语义占比 α
        self.beta_dist: float = float(hp.get("beta_dist", 1.0))   # 数值距离温度
        self.min_alpha: float = float(hp.get("min_alpha", 0.01))  # 最小权重

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

    def _histogram_jaccard(self, local_labels_dict, nb_labels_dict):
        """
        local_labels_dict, nb_labels_dict: {layer_key: label_vector}

        Return Jaccard similarity between aggregated histograms (0~1).
        """

        K = self.n_clusters
        h_local = torch.zeros(K, device=self.device)
        h_nb = torch.zeros(K, device=self.device)

        # 将所有层的 histogram 聚合
        for key, lbl in local_labels_dict.items():
            if key not in nb_labels_dict:
                continue
            lbl_local = lbl.to(self.device).long()
            lbl_nb = nb_labels_dict[key].to(self.device).long()

            h_local += torch.bincount(lbl_local, minlength=K).float()
            h_nb += torch.bincount(lbl_nb, minlength=K).float()

        # 如果没有重叠层 → 语义信息不足 → 返回 1（中性相似度）
        if h_local.sum() == 0 or h_nb.sum() == 0:
            return 1.0

        # 归一化
        h_local = h_local / (h_local.sum() + 1e-8)
        h_nb = h_nb / (h_nb.sum() + 1e-8)

        inter = torch.min(h_local, h_nb).sum()
        union = torch.max(h_local, h_nb).sum() + 1e-8

        return float((inter / union).item())


    # ============================================================
    # 聚合
    # ============================================================
    @torch.no_grad()
    def aggregate(self):
        # 无邻居，直接返回
        if len(self.neighbor_model_weights_buffer) == 0:
            return

        # =========================
        #  首次聚合：还没有 cluster_model
        # =========================
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

            self.model.load_state_dict(avg_state)
            self.cluster_model = self._cluster_and_prune_model_weights()

            # 首次聚合累加所有邻居的 mass
            recv_mass = sum(
                float(tpl[-1].get("ps_mass_share", 0.0))
                for tpl in self.neighbor_model_weights_buffer
                if isinstance(tpl[-1], dict)
            )
            self.ps_mass += float(recv_mass)

            self.neighbor_model_weights_buffer.clear()
            return

        # =========================
        #  常规聚合：已有 cluster_model
        # =========================
        neighbor_states = []
        neighbor_masses = []
        neighbor_cents_list = []
        neighbor_labels_list = []

        for tpl in self.neighbor_model_weights_buffer:
            # 约定：tpl = (state_dict, cents, labels, meta)
            st = tpl[0]
            neighbor_states.append(st)

            meta = tpl[-1] if isinstance(tpl[-1], dict) else {}
            neighbor_masses.append(float(meta.get("ps_mass_share", 0.0)))

            if len(tpl) >= 2 and isinstance(tpl[1], dict):
                neighbor_cents_list.append(tpl[1])
            else:
                neighbor_cents_list.append({})

            if len(tpl) >= 3 and isinstance(tpl[2], dict):
                neighbor_labels_list.append(tpl[2])
            else:
                neighbor_labels_list.append({})

        if len(neighbor_states) == 0:
            self.neighbor_model_weights_buffer.clear()
            return

        keys = set(neighbor_states[0].keys())
        local_mass = float(self.ps_mass)
        local_state = self.model.state_dict()
        local_cents = self.cluster_model[1]  # {layer: [K,1]}
        local_labels = self.cluster_model[2]  # {layer: [num_params]}

        # =======================================================
        #  第一步：为每个邻居算一个“未归一化相似 score_j”
        #          score_j = (J_j + eps)^alpha_mix * (S_j + eps)^(1-alpha_mix)
        # =======================================================
        score_list = []
        if (not self.use_similarity_weight) or (len(local_cents) == 0) or (len(local_labels) == 0):
            # 不用相似度加权时，后面统一设 α_j = 1
            score_list = [1.0 for _ in neighbor_states]
        else:
            for c_nb, lbl_nb in zip(neighbor_cents_list, neighbor_labels_list):

                # ---------- 数值相似 S_j：质心 L2^2 距离的指数衰减 ----------
                numeric_diffs = []
                if isinstance(c_nb, dict) and len(c_nb) > 0:
                    for key, c_local in local_cents.items():
                        if key not in c_nb:
                            continue
                        c1 = c_local.to(self.device)
                        c2 = c_nb[key].to(self.device)
                        if c1.shape != c2.shape:
                            continue
                        diff = (c1 - c2).view(-1)
                        numeric_diffs.append(diff.pow(2).mean())

                if numeric_diffs:
                    dist_mean = torch.stack(numeric_diffs).mean()
                    S_j = torch.exp(- self.beta_dist * dist_mean).item()  # in (0,1]
                else:
                    # 没有质心信息 → 数值上中性
                    S_j = 1.0

                # ---------- 语义相似 J_j：Hard-Jaccard on cluster hist ----------
                if isinstance(lbl_nb, dict) and len(lbl_nb) > 0:
                    J_j = self._histogram_jaccard(local_labels, lbl_nb)  # in [0,1]
                else:
                    # 没有 label 信息 → 语义中性
                    J_j = 1.0

                # ---------- 混合 score_j ----------
                score_j = (J_j + self.sim_eps) ** self.alpha_mix * (S_j + self.sim_eps) ** (1.0 - self.alpha_mix)
                score_list.append(float(score_j))

        # 避免极端情况：全部 score <= 0
        if all(s <= 0.0 for s in score_list):
            score_list = [1.0 for _ in score_list]

        # =======================================================
        #  第二步：归一化成 α_j，使平均 α_j ≈ 1
        #          α_j = score_j / mean(score)
        #          → 相似的 >1，不相似的 <1
        # =======================================================
        mean_score = sum(score_list) / (len(score_list) + self.sim_eps)
        if mean_score <= 0:
            mean_score = 1.0

        eff_masses = []
        eff_cents = []

        for m_raw, c_nb, score_j in zip(
                neighbor_masses, neighbor_cents_list, score_list
        ):
            if not self.use_similarity_weight:
                alpha_j = 1.0
            else:
                alpha_j = score_j / (mean_score + self.sim_eps)  # 平均约为 1

            m_eff = alpha_j * m_raw
            eff_masses.append(m_eff)
            eff_cents.append(c_nb)

        # 若所有邻居有效质量全为 0，退化为本地不聚合
        if sum(eff_masses) == 0.0:
            self.neighbor_model_weights_buffer.clear()
            return

        # =======================================================
        #  push-sum mass 更新（使用 reweighted mass）
        # =======================================================
        total_mass = local_mass + sum(eff_masses)
        if total_mass <= 0:
            total_mass = local_mass if local_mass > 0 else 1.0
        self.ps_mass = float(total_mass)

        # =======================================================
        #  模型聚合：邻居用 m_eff，本地用 local_mass
        # =======================================================
        avg: Dict[str, torch.Tensor] = {}

        for k in keys:
            acc = None

            # 邻居贡献
            for st, m_eff in zip(neighbor_states, eff_masses):
                if m_eff <= 0.0:
                    continue
                if k not in st:
                    continue
                t = st[k].to(self.device)
                acc = t.mul(m_eff) if acc is None else acc.add(t, alpha=m_eff)

            # 本地贡献
            if local_mass > 0.0:
                t_local = local_state[k].to(self.device)
                acc = t_local.mul(local_mass) if acc is None else acc.add(t_local, alpha=local_mass)

            avg[k] = acc.div(total_mass)

        self.model.load_state_dict(avg)

        # =======================================================
        #  global_cents 更新：也用 m_eff 做权重
        # =======================================================
        new_global_cents: Dict[str, torch.Tensor] = {}
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
