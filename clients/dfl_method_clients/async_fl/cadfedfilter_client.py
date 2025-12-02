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

    核心新点：版本 A — “更新方向与全局趋势的一致性” q_align
    - 用质心的更新方向 ΔC_i 与 “朝全局质心移动的方向” d_target 的夹角
    - q_align 越大，说明本轮更新越是“顺着全局趋势走”，质量越高
    - 在 gossip 聚合时，用 q_align 对邻居的 ps_mass 做 reweight
    """

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

        # Push-sum mass
        self.ps_mass: float = 1.0

        # 是否使用全局质心做 KMeans 初始化 + 作为“全局趋势”参考
        self.use_global_cents: bool = bool(hp.get('use_global_cents', True))
        self.global_cents: Dict[str, torch.Tensor] = {}

        # ============ 更新方向一致性 q_align ============
        self.use_align_quality: bool = bool(hp.get("use_align_quality", True))
        # 对 cos 对齐度的幂次，调节映射形状（>1 更尖锐，=1 线性）
        self.align_gamma: float = float(hp.get("align_gamma", 1.0))
        # cos 计算用的 eps
        self.align_eps: float = float(hp.get("align_eps", 1e-8))
        # 对齐质量的下界，避免完全灭掉某些邻居
        self.align_min_score: float = float(hp.get("align_min_score", 0.0))

        # 最近一轮训练得到的对齐质量 q_align \in [0,1]
        self.align_score: float = 1.0
        # 存本地质心快照（训练前）
        self._prev_local_cents: Dict[str, torch.Tensor] = {}
        # 存对齐目标质心（global snapshot）
        self._align_target_cents: Dict[str, torch.Tensor] = {}

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
        """
        对当前 self.model 做 KMeans 质心压缩 + 掩码剪枝
        返回:
          - clustered: 压缩后权重
          - cents: {layer: [K,1]}
          - labels: {layer: [num_params]}
        """
        clustered: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}
        cents: Dict[str, torch.Tensor] = {}
        labels: Dict[str, torch.Tensor] = {}

        state = self.model.state_dict()
        for key, w in state.items():
            if "weight" in key and "bn" not in key and "downsample" not in key:
                orig = w.shape
                flat = w.detach().view(-1, 1)

                kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)

                # 如果有全局质心，作为 KMeans 初始
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
                cent_sorted = cent_sorted.to(self.device)

                new_w = cent_sorted[lab_sorted].view(orig)
                is_zero = cent_sorted.view(-1) == 0
                m = (is_zero[lab_sorted].view(orig) == 0)

                clustered[key] = new_w
                mask[key] = m.bool()
                cents[key] = cent_sorted
                labels[key] = lab_sorted.view(-1).to(self.device)
            else:
                # 非聚类层：原样保留
                clustered[key] = w.detach().to(self.device)
                mask[key] = torch.ones_like(w, dtype=torch.bool, device=w.device)

        self.mask = mask

        # 若尚未有 global_cents，用本地的形状初始化
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
    # 版本 A：对齐快照 & q_align 计算
    # ============================================================
    def _capture_align_snapshot(self):
        """
        在本轮本地训练开始前调用：
        - 确保 self.cluster_model 对应当前模型
        - 保存当前本地质心 C_old
        - 保存对齐目标质心 G_target（当前 global_cents 的快照）
        """
        if self.cluster_model is None:
            self.cluster_model = self._cluster_and_prune_model_weights()

        _, local_cents, _ = self.cluster_model

        self._prev_local_cents = {
            k: v.detach().clone().to(self.device) for k, v in local_cents.items()
        }

        if self.use_global_cents and len(self.global_cents) > 0:
            self._align_target_cents = {
                k: v.detach().clone().to(self.device) for k, v in self.global_cents.items()
            }
        else:
            # 没有全局信息时，用自己 C_old 做 target → 对齐分数退化为 1（中性）
            self._align_target_cents = {
                k: v.detach().clone().to(self.device) for k, v in local_cents.items()
            }

    def _update_align_score(self):
        """
        在本轮本地训练 + 重新聚类之后调用：
        - 使用 C_new, C_old, G_target 计算更新方向与目标方向的对齐度 q_align ∈ [0,1]
        公式：
            ΔC = C_new - C_old
            d_target = G_target - C_old
            cos = max(0, <ΔC, d_target> / (||ΔC||·||d_target||))
            q_align = max(align_min_score, cos^align_gamma)
        """
        if (not self.use_align_quality) or (self.cluster_model is None):
            self.align_score = 1.0
            return

        if (len(self._prev_local_cents) == 0) or (len(self._align_target_cents) == 0):
            self.align_score = 1.0
            return

        _, local_cents_new, _ = self.cluster_model

        delta_local_list = []
        delta_target_list = []

        for key, c_prev in self._prev_local_cents.items():
            if key not in local_cents_new or key not in self._align_target_cents:
                continue

            c_prev = c_prev.to(self.device)
            c_new = local_cents_new[key].to(self.device)
            c_tgt = self._align_target_cents[key].to(self.device)

            if c_prev.shape != c_new.shape or c_prev.shape != c_tgt.shape:
                continue

            delta_local_list.append((c_new - c_prev).view(-1))
            delta_target_list.append((c_tgt - c_prev).view(-1))

        if not delta_local_list or not delta_target_list:
            # 没有可用信息，则保持中性
            self.align_score = 1.0
            return

        delta_local = torch.cat(delta_local_list, dim=0)
        delta_target = torch.cat(delta_target_list, dim=0)

        norm_local = torch.norm(delta_local)
        norm_target = torch.norm(delta_target)
        eps = self.align_eps

        if norm_local <= eps or norm_target <= eps:
            self.align_score = 1.0
            return

        cos = torch.dot(delta_local, delta_target) / (norm_local * norm_target + eps)
        cos_clamped = torch.clamp(cos, min=0.0, max=1.0).item()

        gamma = self.align_gamma
        q = (cos_clamped ** gamma) if gamma != 1.0 else cos_clamped
        q = max(self.align_min_score, float(q))

        self.align_score = q

    # ============================================================
    # 模型重构（解码压缩）
    # ============================================================
    def _reconstruct_from_compressed(
        self,
        cents: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        uncompressed: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        reconstructed: Dict[str, torch.Tensor] = {}

        # 非压缩层先写入
        for k, v in uncompressed.items():
            reconstructed[k] = v.to(self.device)

        local_state = self.model.state_dict()

        # 压缩层：质心查表 + reshape
        for key, idx in labels.items():
            if key not in cents:
                continue

            c = cents[key].to(self.device)          # [K,1]
            idx = idx.to(self.device).long()        # [num_params]

            if key not in local_state:
                continue

            orig_shape = local_state[key].shape
            w_recon = c[idx].view(orig_shape)
            reconstructed[key] = w_recon

        return reconstructed

    # ============================================================
    # Push-sum meta 打包（带 q_align）
    # ============================================================
    def _prepare_maturity_meta(self) -> Dict[str, Any]:
        """
        标准 push-sum 质量拆分：
          - 把当前 ps_mass 均分成 (k_push + 1) 份，一份留给自己，其余发给邻居
        新增：
          - 把本轮的对齐质量 q_align 一起打包（不直接缩放 share，而是在接收端使用）
        """
        d_out = self.k_push + 1
        share = float(self.ps_mass) / float(d_out)
        self.ps_mass = share

        q_align = float(self.align_score) if self.use_align_quality else 1.0

        return {
            'version_meta': 'cadfedfilter_meta_v1',
            'sender_id': self.id,
            'version': self.local_version,
            'sender_time': self.last_update_time,
            "ps_mass_share": share,
            "q_align": q_align,
        }

    # ============================================================
    # 聚合：标准 push-sum + 对齐质量 reweight
    # ============================================================
    @torch.no_grad()
    def aggregate(self):

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

            self.model.load_state_dict(avg_state)
            self.cluster_model = self._cluster_and_prune_model_weights()

            recv_mass = sum(
                float(tpl[-1].get("ps_mass_share", 0.0))
                for tpl in self.neighbor_model_weights_buffer
                if isinstance(tpl[-1], dict)
            )
            self.ps_mass += float(recv_mass)

            self.neighbor_model_weights_buffer.clear()
            return

        # ---------- 常规聚合 ----------
        neighbor_states = []
        neighbor_masses_raw = []
        neighbor_cents_list = []

        neighbor_qalign_list = []

        for tpl in self.neighbor_model_weights_buffer:
            st = tpl[0]
            neighbor_states.append(st)

            meta = tpl[-1] if isinstance(tpl[-1], dict) else {}
            neighbor_masses_raw.append(float(meta.get("ps_mass_share", 0.0)))
            neighbor_qalign_list.append(float(meta.get("q_align", 1.0)))

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

        # ---------- 使用 q_align 对邻居 mass 进行 reweight ----------
        # 我们希望：相对质量由 q_align 决定，但整体平均 scale 约为 1
        # α_j = q_align_j / mean(q_align)
        if self.use_align_quality:
            scores = [max(0.0, q) for q in neighbor_qalign_list]
            if all(s <= 0.0 for s in scores):
                scores = [1.0 for _ in scores]

            mean_score = sum(scores) / (len(scores) + 1e-8)
            if mean_score <= 0:
                mean_score = 1.0

            eff_masses = []
            for m_raw, s in zip(neighbor_masses_raw, scores):
                alpha_j = s / (mean_score + 1e-8)
                alpha_j = max(self.align_min_score, float(alpha_j))
                eff_masses.append(alpha_j * m_raw)
        else:
            eff_masses = neighbor_masses_raw

        # 若所有邻居有效质量为 0，则退化为本地不聚合
        if sum(eff_masses) == 0.0:
            self.neighbor_model_weights_buffer.clear()
            return

        # ---------- push-sum mass 更新 ----------
        total_mass = local_mass + sum(eff_masses)
        if total_mass <= 0:
            total_mass = local_mass if local_mass > 0 else 1.0
        self.ps_mass = float(total_mass)

        # ---------- 模型聚合 ----------
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

        # ---------- 更新 global_cents（用 reweighted mass） ----------
        if self.use_global_cents and self.cluster_model is not None:
            _, local_cents, _ = self.cluster_model
            new_global_cents: Dict[str, torch.Tensor] = {}

            for key, c_local in local_cents.items():
                c_local = c_local.to(self.device)
                num = c_local * local_mass
                den = local_mass

                for c_nb, m_eff in zip(neighbor_cents_list, eff_masses):
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
    # 训练接口：在 train() 里处理对齐快照 + q_align
    # ============================================================
    def train(self):
        """
        一轮本地训练：
          1) 若 cluster_model 为空，先根据当前模型建立初始质心
          2) 记录 C_old 和 G_target 作为对齐快照
          3) 本地 SGD 训练
          4) 重新聚类得到 C_new
          5) 计算 q_align（更新方向 vs 全局目标方向的一致性）
        """
        # 1 + 2: 对齐快照
        self._capture_align_snapshot()

        # 3: 本地训练
        self._local_train()

        # 4: 用新模型重新聚类
        self.cluster_model = self._cluster_and_prune_model_weights()

        # 5: 计算 q_align
        self._update_align_score()

    # ============================================================
    # 模型发送 & 接收（压缩 + meta）
    # ============================================================
    def set_init_model(self, model: nn.Module):
        self.model = deepcopy(model).to(self.device)
        self.cluster_model = None
        self.ps_mass = 1.0
        self.global_cents = {}
        self.align_score = 1.0
        self._prev_local_cents.clear()
        self._align_target_cents.clear()

    def send_model(self):
        """
        发送压缩模型：
          payload = {cents, labels, uncompressed}
          meta    = {ps_mass_share, q_align, ...}
        """
        if self.cluster_model is None:
            clustered_state_dict, cents, labels = self._cluster_and_prune_model_weights()
            self.cluster_model = (clustered_state_dict, cents, labels)
        else:
            clustered_state_dict, cents, labels = self.cluster_model

        # 未压缩层
        uncompressed = {
            k: v for k, v in clustered_state_dict.items() if k not in cents
        }

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
