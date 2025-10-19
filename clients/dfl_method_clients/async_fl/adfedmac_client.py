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

#
# class ADFedMACClient(Client):
#     """
#     混合策略（不修改协调器、无补发）：
#
#     - 普通客户端（非延迟）：
#         * 缓冲邻居的 state_dict；
#         * 训练前（或 set_init_model 当刻）做一次等权平均： self + neighbors；
#         * 训练使用基类 _local_train()；
#         * 不进行结构对齐。
#     - 延迟客户端：
#         * 预训练阶段：行为与上面“普通客户端”一致（可用等权聚合 + _local_train()），不做对齐；
#         * 预训练结束：切换到结构对齐模式（DKM + 成熟度/CFD），此后不再做直接平均聚合。
#     """
#
#     # ===================== 初始化 & 策略开关 =====================
#     def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
#         hp = dict(hyperparam)
#
#         self.is_delayed: bool = hp.get('is_delayed', False)
#
#         # 预训练（延迟客户端专用；单位=burst）
#         self.warmup_bursts: int = int(hp.get('warmup_bursts', 1))
#         self._warmup_done: int = 0
#
#         # 与 AsyncDFedAvgClient 一致：不强制“收到即融合”，聚合时机由上层调度或 train() 前触发
#         self.fuse_on_receive: bool = bool(hp.get('fuse_on_receive', False))
#
#         # 对齐相关（仅延迟客户端在预训练后启用）
#         self.lambda_alignment: float = float(hp.get('lambda_alignment', 0.01))
#         self.n_clusters: int = int(hp.get('n_clusters', 16))
#         self.teacher_gamma: float = float(hp.get('teacher_gamma', 1.0))
#         self.teacher_blend: float = float(hp.get('teacher_blend', 0.6))
#         self.maturity_window: int = int(hp.get('maturity_window', 5))
#         self.beta_drift: float = float(hp.get('beta_drift', 1.0))
#         self.beta_invar: float = float(hp.get('beta_invar', 1.0))
#         self.beta_mask: float = float(hp.get('beta_mask', 0.5))
#         self.maturity_eps: float = float(hp.get('maturity_eps', 1e-8))
#
#         super().__init__(client_id, dataset_index, full_dataset, hp, device)
#
#         # --- 运行期容器 ---
#         # 用于“聚合”的邻居缓存（存 state_dict），普通 & 延迟-预训练 共用
#         self._agg_buffer: List[Dict[str, torch.Tensor]] = []
#
#         # 用于“对齐”的邻居缓存（存 3/4 元组：(clustered_state, centroids, labels [,meta])），仅延迟-对齐使用
#         self.neighbor_model_weights: List[Any] = []
#
#         # 对齐组件
#         self.teacher_info_list: List[Dict[str, Any]] = []
#         self.dkm_layers: Dict[str, MultiTeacherDKMLayer] = {}
#         self.mask: Dict[str, torch.Tensor] = {}
#         self._local_hist = defaultdict(lambda: deque(maxlen=self.maturity_window))
#         self._local_mask_hist = defaultdict(lambda: deque(maxlen=2))
#
#         # 训练后缓存（避免重复聚类）
#         self.cluster_model: Optional[
#             Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
#         ] = None
#
#     # ===================== 与 AsyncDFedAvg 对齐的聚合（等权：self+neighbors） =====================
#     @torch.no_grad()
#     def _aggregate_dfedavg_style_(self, neighbor_states: List[Dict[str, torch.Tensor]]):
#         if not neighbor_states:
#             return
#         current = {k: v.detach().clone().to(self.device) for k, v in self.model.state_dict().items()}
#         count = 1 + len(neighbor_states)
#         for sd in neighbor_states:
#             for k in current.keys():
#                 if k in sd and sd[k].shape == current[k].shape:
#                     current[k] += sd[k].to(self.device)
#         for k in current.keys():
#             current[k] /= float(count)
#         self.model.load_state_dict(current)
#
#     # ===================== DKM/聚类/成熟度（与前版一致，略微精简） =====================
#     def _register_dkm_layers(self) -> None:
#         self.dkm_layers = {}
#         for key in self.model.state_dict().keys():
#             if 'weight' in key and 'bn' not in key and 'downsample' not in key and 'conv' in key:
#                 self.dkm_layers[key] = MultiTeacherDKMLayer(
#                     n_clusters=self.n_clusters, alpha_mix=0.7, beta_dist=2.0
#                 ).to(self.device)
#
#     def _prune_model_weights(self):
#         pruned_state_dict = {}
#         for key, weight in self.model.state_dict().items():
#             if key in self.mask:
#                 pruned_state_dict[key] = weight * self.mask[key]
#             else:
#                 pruned_state_dict[key] = weight
#         return pruned_state_dict
#
#     @staticmethod
#     def _sort_centroids_and_remap(c1d: torch.Tensor, lbl: torch.Tensor):
#         c = c1d.view(-1).detach()
#         order = torch.argsort(c)
#         sorted_c = c[order].view(-1, 1)
#         old2new = torch.empty_like(order)
#         old2new[order] = torch.arange(order.numel(), device=order.device)
#         new_labels = old2new[lbl]
#         return sorted_c, new_labels
#
#     @staticmethod
#     def _cluster_intra_var(flat_w: torch.Tensor, labels: torch.Tensor, K: int, eps: float = 1e-8) -> torch.Tensor:
#         vars_out = flat_w.new_zeros(K, 1)
#         for k in range(K):
#             idx = (labels == k)
#             if idx.any():
#                 w = flat_w[idx].view(-1)
#                 vars_out[k, 0] = torch.var(w, unbiased=False)
#             else:
#                 vars_out[k, 0] = eps
#         return vars_out + eps
#
#     def _update_local_history(self, centroids: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]):
#         for layer, c in centroids.items():
#             self._local_hist[layer].append(c.detach().cpu())
#         for layer, m in mask.items():
#             self._local_mask_hist[layer].append(m.detach().cpu())
#
#     def _cluster_and_prune_model_weights(self):
#         clustered: Dict[str, torch.Tensor] = {}
#         mask: Dict[str, torch.Tensor] = {}
#         cents: Dict[str, torch.Tensor] = {}
#         labels: Dict[str, torch.Tensor] = {}
#
#         state = self.model.state_dict()
#         for key, w in state.items():
#             if 'weight' in key and 'bn' not in key and 'downsample' not in key:
#                 orig = w.shape
#                 kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)
#                 flat = w.detach().view(-1, 1)
#                 kmeans.fit(flat)
#                 cent_sorted, lab_sorted = self._sort_centroids_and_remap(
#                     kmeans.centroids.view(-1, 1), kmeans.labels_)
#                 new_w = cent_sorted[lab_sorted].view(orig)
#                 is_zero = (cent_sorted.view(-1) == 0)
#                 m = (is_zero[lab_sorted].view(orig) == 0)
#
#                 clustered[key] = new_w
#                 mask[key] = m.bool()
#                 cents[key] = cent_sorted.to(self.device)
#                 labels[key] = lab_sorted.view(-1).to(self.device)
#             else:
#                 clustered[key] = w
#                 mask[key] = torch.ones_like(w, dtype=torch.bool)
#
#         self.mask = mask
#         self._update_local_history(cents, mask)
#         return clustered, cents, labels
#
#     def _c_swag_precision(self, hist: deque) -> torch.Tensor:
#         if len(hist) < 2:
#             return hist[-1].new_ones(hist[-1].shape[0], 1)
#         stack = torch.stack(list(hist), dim=0)
#         var = torch.var(stack, dim=0, unbiased=False)
#         return 1.0 / (var + self.maturity_eps)
#
#     def _stability_scores(self, layer_key: str,
#                           cents_now: torch.Tensor,
#                           labels_now: torch.Tensor,
#                           mask_now: torch.Tensor) -> torch.Tensor:
#         dev = cents_now.device
#         if len(self._local_hist[layer_key]) >= 1:
#             prev_c = self._local_hist[layer_key][-1].to(dev)
#             drift = torch.abs(cents_now - prev_c)
#         else:
#             drift = torch.zeros_like(cents_now)
#
#         # flat_w = self.model.state_dict()[layer_key].to(dev).view(-1, 1).detach()
#         # K = cents_now.shape[0]
#         # invar = self._cluster_intra_var(flat_w, labels_now, K, eps=self.maturity_eps)
#
#         if len(self._local_mask_hist[layer_key]) >= 1:
#             prev_m = self._local_mask_hist[layer_key][-1].to(dev)
#             flips = (prev_m ^ mask_now).float().mean()
#         else:
#             flips = torch.tensor(0.0, device=dev)
#
#         #  y_{i} = e^{x_{i}}
#         # stab = torch.exp(- self.beta_drift * drift - self.beta_invar * invar - self.beta_mask * flips)
#         stab = torch.exp(- self.beta_drift * drift  - self.beta_mask * flips)
#
#         return stab.clamp_min(1e-6)
#
#     def _local_maturity(self, layer_key: str,
#                         cents_now: torch.Tensor,
#                         labels_now: torch.Tensor,
#                         mask_now: torch.Tensor) -> torch.Tensor:
#         lam = self._c_swag_precision(self._local_hist[layer_key]).to(cents_now.device)
#         stab = self._stability_scores(layer_key, cents_now, labels_now, mask_now)
#         return lam * stab
#
#     def _prepare_maturity_meta(self, cents: Dict[str, torch.Tensor],
#                                labels: Dict[str, torch.Tensor]) -> Dict[str, Any]:
#         layer_maturity = {}
#         for layer_key, c_now in cents.items():
#             m_now = self.mask[layer_key].to(c_now.device)
#             l_now = labels[layer_key]
#             maturity_vec = self._local_maturity(layer_key, c_now, l_now, m_now)
#             layer_maturity[layer_key] = float(maturity_vec.mean().item())
#         return {
#             'version_meta': 'dfedmac_meta_v1',
#             'sender_id': self.id,
#             'version': int(self.local_version),
#             'sender_time': float(self.last_update_time),
#             'layer_maturity': layer_maturity
#         }
#
#     def _all_teacher_info(self) -> None:
#         if self.cluster_model is None:
#             return
#         else:
#             local_cents = self.cluster_model[1]
#
#         t_cents_list: List[Dict[str, torch.Tensor]] = []
#         t_meta_list: List[Optional[Dict[str, Any]]] = []
#         cfd_matrix: List[List[float]] = []
#
#         for item in self.neighbor_model_weights:
#             if isinstance(item, (list, tuple)) and (len(item) == 4):
#                 _, c_t, _, meta = item
#                 t_meta_list.append(meta)
#             elif isinstance(item, (list, tuple)) and (len(item) == 3):
#                 _, c_t, _ = item
#                 meta = None
#                 t_meta_list.append(meta)
#             else:
#                 continue
#             t_cents_list.append(c_t)
#
#             per_layer = []
#             for lk in local_cents:
#                 per_layer.append(_cfd_distance(local_cents[lk].detach().float(),
#                                                c_t[lk].detach().float()))
#             cfd_matrix.append(per_layer)
#
#         if len(t_cents_list) == 0:
#             self.teacher_info_list = []
#             return
#
#         cfd_tensor = torch.tensor(cfd_matrix, dtype=torch.float, device=self.device)  # [T,L]
#         cfd_scores = torch.mean(cfd_tensor, dim=1)  # [T]
#         min_v, max_v = cfd_scores.min(), cfd_scores.max()
#         normed = (cfd_scores - min_v) / (max_v - min_v + 1e-8)
#         alpha_base = torch.softmax(-2.0 * normed, dim=0)
#
#         layer_keys = list(local_cents.keys())
#         T = len(t_cents_list)
#         maturity_mat = torch.ones(len(layer_keys), T, dtype=torch.float, device=self.device)
#         for t_idx, meta in enumerate(t_meta_list):
#             if isinstance(meta, dict) and ('layer_maturity' in meta):
#                 lm = meta['layer_maturity']
#                 for li, lk in enumerate(layer_keys):
#                     if lk in lm:
#                         maturity_mat[li, t_idx] = max(float(lm[lk]), self.maturity_eps)
#
#         vmin = maturity_mat.min(dim=1, keepdim=True).values
#         vmax = maturity_mat.max(dim=1, keepdim=True).values
#         maturity_norm = torch.clamp((maturity_mat - vmin) / (vmax - vmin + 1e-8),
#                                     self.maturity_eps, 1.0)
#
#         self.teacher_info_list = []
#         for t in range(T):
#             layer_maturity_dict = {layer_keys[li]: float(maturity_norm[li, t].item())
#                                    for li in range(len(layer_keys))}
#             self.teacher_info_list.append({
#                 'centroids': t_cents_list[t],
#                 'alpha': float(alpha_base[t].item()),
#                 'layer_maturity': layer_maturity_dict
#             })
#
#     def _compute_alignment_loss(self) -> torch.Tensor:
#         if len(self.teacher_info_list) == 0 or len(self.dkm_layers) == 0 or self.lambda_alignment == 0.0:
#             return torch.zeros((), device=self.device)
#
#         losses = []
#         st = self.model.state_dict()
#         for lk, dkm in self.dkm_layers.items():
#             Wf = st[lk].to(self.device).view(-1, 1)
#             t_cents = torch.stack([t['centroids'][lk].to(self.device)
#                                    for t in self.teacher_info_list], dim=0)
#             maturity = torch.tensor([t['layer_maturity'].get(lk, 0.0)
#                                      for t in self.teacher_info_list],
#                                     device=self.device, dtype=torch.float)
#             alpha_base = torch.tensor([t['alpha'] for t in self.teacher_info_list],
#                                       device=self.device, dtype=torch.float)
#             alpha_base = alpha_base / (alpha_base.sum() + 1e-12)
#
#             alpha_eff_raw = alpha_base * (maturity ** self.teacher_gamma)
#             alpha_eff = (alpha_eff_raw / alpha_eff_raw.sum()
#                          if alpha_eff_raw.sum() > 1e-12
#                          else torch.ones_like(alpha_eff_raw) / max(1, alpha_eff_raw.numel()))
#
#             alpha_eff = (1.0 - self.teacher_blend) * alpha_base + self.teacher_blend * alpha_eff
#             alpha_eff = alpha_eff / (alpha_eff.sum() + 1e-12)
#
#             X_rec, _, _ = dkm(
#                 Wf,
#                 teacher_centroids=t_cents,
#                 teacher_alphas=alpha_eff,
#                 teacher_index_tables=None,
#                 lambda_teacher=1.0
#             )
#             losses.append(F.mse_loss(Wf, X_rec))
#
#         return torch.stack(losses).sum() if losses else torch.zeros((), device=self.device)
#
#     # ===================== 接收邻居 =====================
#     def receive_neighbor_model(self, neighbor_model: Any):
#         """
#         - 预训练阶段（延迟）或普通客户端：只接收可聚合的 state_dict（若为3/4元组则取第0项）。
#         - 对齐阶段（延迟）：只接收 3/4 元组用于老师集合；忽略纯 state_dict。
#         - 不在此处强制“收到即融合”，保持与 AsyncDFedAvgClient 接近；是否融合由 fuse_on_receive 决定。
#         """
#         in_warmup = self.is_delayed and (self._warmup_done < self.warmup_bursts)
#
#         # —— 可聚合阶段（普通 & 延迟预训练）——
#         if (not self.is_delayed) or in_warmup:
#             cand_sd = None
#             if isinstance(neighbor_model, (list, tuple)) and len(neighbor_model) >= 1:
#                 cand_sd = neighbor_model[0]           # 取 clustered_state_dict
#             elif isinstance(neighbor_model, dict):
#                 cand_sd = neighbor_model              # 纯 state_dict
#             if isinstance(cand_sd, dict):
#                 self._agg_buffer.append(cand_sd)
#                 if self.fuse_on_receive:
#                     self._aggregate_dfedavg_style_(self._agg_buffer)
#                     self._agg_buffer.clear()
#             return
#
#         # —— 对齐阶段（延迟）—— 仅缓存3/4元组
#         if isinstance(neighbor_model, (list, tuple)) and (len(neighbor_model) in (3, 4)):
#             self.neighbor_model_weights.append(neighbor_model)
#             if self.buffer_limit and self.buffer_limit > 0:
#                 overflow = len(self.neighbor_model_weights) - self.buffer_limit
#                 if overflow > 0:
#                     self.neighbor_model_weights = self.neighbor_model_weights[overflow:]
#         # 纯 state_dict 在对齐阶段忽略
#
#     # ===================== 聚合（与 AsyncDFedAvg 风格一致） =====================
#     def aggregate(self):
#         """
#         与 AsyncDFedAvgClient 一致：只要缓冲里有邻居，就做一次等权平均（self+neighbors），然后清空缓冲。
#         何时触发由协调器或 train() 前逻辑决定；不强制自动触发。
#         """
#         if len(self._agg_buffer) == 0:
#             return
#         self._aggregate_dfedavg_style_(self._agg_buffer)
#         self._agg_buffer.clear()
#
#
#     def _local_train(self):
#         """默认本地训练：跑 self.epochs 个 epoch。子类的 train() 可直接调用它。"""
#         if self.client_train_loader is None:
#             raise RuntimeError("DataLoader 尚未初始化：请先在 Join 时调用 init_client()。")
#         self.model.train()
#         for _ in range(self.epochs):
#             for x, labels in self.client_train_loader:
#                 self.model.load_state_dict(self._prune_model_weights())
#                 x, labels = x.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad(set_to_none=True)
#                 outputs = self.model(x)
#                 loss = self.criterion(outputs, labels).mean()
#                 loss.backward()
#                 self.optimizer.step()
#
#     # ===================== 训练 =====================
#     def train(self):
#         """
#         - 普通客户端：训练前可调用 aggregate()（由上层决定），训练本体走 _local_train()。
#         - 延迟客户端：
#             * 预训练阶段：与普通客户端一致（可聚合 + _local_train()）；
#             * 预训练结束：切换对齐模式（监督 + 对齐损失）。
#         """
#         in_warmup = self.is_delayed and (self._warmup_done < self.warmup_bursts)
#
#         # 若未收到即融合，训练前可由上层/调度调用 aggregate()；这里不强制调用以贴合 AsyncDFedAvgClient。
#         if (not self.fuse_on_receive) and ((not self.is_delayed) or in_warmup):
#             self.aggregate()
#
#         # —— 预训练阶段 or 普通客户端：仅监督训练 —— #
#         if (not self.is_delayed) or in_warmup:
#             self._local_train()
#             # 延迟：完成一个 burst 计数
#             if in_warmup:
#                 self._warmup_done += 1
#             # 训练后准备对外通信的结构化载荷（与对齐阶段保持一致的接口）
#             self.cluster_model = self._cluster_and_prune_model_weights()
#             return
#
#         # —— 对齐阶段（延迟）：监督 + 对齐 —— #
#         if len(self.dkm_layers) == 0:
#             self._register_dkm_layers()
#         self._all_teacher_info()
#
#         if self.client_train_loader is None:
#             raise RuntimeError("DataLoader not initialized; ensure init_client() on join.")
#
#         self.model.train()
#         for _ in range(self.epochs):
#             for x, labels in self.client_train_loader:
#                 self.model.load_state_dict(self._prune_model_weights())
#                 x, labels = x.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad(set_to_none=True)
#                 outputs = self.model(x)
#                 loss_sup = self.criterion(outputs, labels).mean()
#                 loss_align = self._compute_alignment_loss()
#                 loss = loss_sup + self.lambda_alignment * loss_align
#                 loss.backward()
#                 self.optimizer.step()
#
#         # 训练后：更新结构化表示
#         self.cluster_model = self._cluster_and_prune_model_weights()
#         # 老师缓冲用于下一 burst，按需清理
#         self.neighbor_model_weights.clear()
#
#     # ===================== 初始化模型 =====================
#     def set_init_model(self, model: torch.nn.Module):
#         """
#         与 AsyncDFedAvgClient 对齐：JOIN 当刻若聚合缓冲已有邻居模型，可先平均一次再开始训练。
#         """
#         self.model = deepcopy(model).to(self.device)
#         self.cluster_model = None
#         self.teacher_info_list.clear()
#         self.mask.clear()
#         self._warmup_done = 0
#
#         # 若此刻已有邻居（无补发时通常为空，但保持一致的语义）
#         if len(self._agg_buffer) != 0:
#             self._aggregate_dfedavg_style_(self._agg_buffer)
#             self._agg_buffer.clear()
#
#     # ===================== 发送（4元组） =====================
#     def send_model(self):
#         """
#         返回 (clustered_state_dict, centroids_dict, labels_dict, meta)：
#         - 普通/预训练阶段：我们也给出结构化版本（从当前模型做一次聚类），方便对端（对齐端）直接使用；
#         - 对齐阶段：直接复用训练后缓存的 cluster_model。
#         """
#         if self.cluster_model is None:
#             clustered_state_dict, cents, labels = self._cluster_and_prune_model_weights()
#         else:
#             clustered_state_dict, cents, labels = self.cluster_model
#         meta = self._prepare_maturity_meta(cents, labels)
#         return clustered_state_dict, cents, labels, meta

class ADFedMACClient(Client):
    """
    ADFedMAC 客户端的简化重构版。
    行为由 `lambda_alignment` 参数唯一控制：
    - 等于 0: 行为与 AsyncDFedAvgClient 完全一致 (聚合 + 纯监督训练)。
    - 大于 0: 启用异步结构化对齐 (CFD + 成熟度 + DKM)。
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        # --- 唯一控制开关 ---
        self.lambda_alignment: float = float(hyperparam.get('lambda_alignment', 0.0))

        # --- 仅在对齐模式下使用的超参数和状态 ---
        if self.lambda_alignment > 0.0:
            self.n_clusters: int = int(hyperparam.get('n_clusters', 16))
            self.teacher_gamma: float = float(hyperparam.get('teacher_gamma', 1.0))
            self.teacher_blend: float = float(hyperparam.get('teacher_blend', 0.6))
            self.maturity_window: int = int(hyperparam.get('maturity_window', 5))
            self.beta_drift: float = float(hyperparam.get('beta_drift', 1.0))
            self.beta_mask: float = float(hyperparam.get('beta_mask', 0.5))
            self.maturity_eps: float = float(hyperparam.get('maturity_eps', 1e-8))

            self.dkm_layers: Dict[str, MultiTeacherDKMLayer] = {}
            self.mask: Dict[str, torch.Tensor] = {}
            self.teacher_info_list: List[Dict[str, Any]] = []
            self._local_hist = defaultdict(lambda: deque(maxlen=self.maturity_window))
            self._local_mask_hist = defaultdict(lambda: deque(maxlen=2))
            self.cluster_model: Optional[Tuple[Dict, Dict, Dict]] = None

    # ===================================================================
    # 1. 公共 API (由协调器调用)
    # ===================================================================
    def set_init_model(self, model: torch.nn.Module):
        """设置初始模型，并重置所有特定于模式的状态。"""
        self.model = deepcopy(model).to(self.device)
        # 总是重置对齐状态，确保模式切换时是干净的
        if hasattr(self, 'cluster_model'):
            self.cluster_model = None
            self.mask.clear()
            self.teacher_info_list.clear()

    def receive_neighbor_model(self, neighbor_model: Any):
        """根据当前模式接收并缓冲邻居模型。"""
        # --- 对齐模式：只接受四元组 ---
        if self.lambda_alignment > 0.0:
            if isinstance(neighbor_model, (list, tuple)) and len(neighbor_model) >= 3:
                self.neighbor_model_weights.append(neighbor_model)
        # --- 非对齐模式：调用基类方法，它能正确处理 state_dict ---
        else:
            super().receive_neighbor_model(neighbor_model)

    def train(self):
        """执行一次本地训练，根据模式选择路径。"""
        if self.lambda_alignment == 0.0:
            self._train_fedavg()
        else:
            self._train_adfedmac()

    def send_model(self) -> Any:
        """生成要发送的载荷，根据模式选择格式。"""
        if self.lambda_alignment == 0.0:
            # 发送原始 state_dict
            return super().send_model()
        else:
            # 发送四元组
            if self.cluster_model is None:
                clustered_state, cents, labels = self._cluster_and_prune_model_weights()
            else:
                clustered_state, cents, labels = self.cluster_model

            meta = self._prepare_maturity_meta(cents, labels)
            return clustered_state, cents, labels, meta

    # ===================================================================
    # 2. FedAvg 核心逻辑 (当 lambda_alignment == 0)
    # ===================================================================
    def aggregate(self):
        """
        等权平均聚合（self + neighbors）。此方法仅在非对齐模式下有意义。
        与 AsyncDFedAvgClient 的实现完全一致。
        """
        if len(self.neighbor_model_weights) == 0:
            return

        current = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        count = 1 + len(self.neighbor_model_weights)

        for sd in self.neighbor_model_weights:
            for k in current.keys():
                current[k] += sd[k].to(self.device)

        for k in current.keys():
            current[k] /= float(count)

        self.model.load_state_dict(current)
        self.neighbor_model_weights.clear()

    def _train_fedavg(self):
        """执行一次标准的 FedAvg 训练：先聚合，再本地训练。"""
        if not self.fuse_on_receive:
            self.aggregate()
        # 调用基类中最纯净的训练方法，确保没有掩码操作
        super()._local_train()

    # ===================================================================
    # 3. ADFedMAC 核心逻辑 (当 lambda_alignment > 0)
    # ===================================================================
    def _train_adfedmac(self):
        """执行一次包含对齐的训练。"""
        if not self.dkm_layers: self._register_dkm_layers()
        self._prepare_teacher_info()

        self.model.train()
        for _ in range(self.epochs):
            for x, labels in self.client_train_loader:
                # 步骤 1: 应用剪枝掩码
                self.model.load_state_dict(self._prune_model_weights())

                # 步骤 2: 计算联合损失
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(x)
                loss_sup = self.criterion(outputs, labels).mean()
                loss_align = self._compute_alignment_loss()
                loss = loss_sup + self.lambda_alignment * loss_align

                # 步骤 3: 反向传播和优化
                loss.backward()
                self.optimizer.step()

        # 训练后更新自身结构化表示，并清空教师缓冲
        self.cluster_model = self._cluster_and_prune_model_weights()
        self.neighbor_model_weights.clear()

    def _prepare_teacher_info(self):
        """从邻居缓冲中构建教师集合，计算相似度和成熟度。"""
        # (此部分内部逻辑不变，代码省略以保持简洁)
        if self.cluster_model is None:
            _, local_cents, _ = self._cluster_and_prune_model_weights()
        else:
            local_cents = self.cluster_model[1]
        teachers = [item for item in self.neighbor_model_weights if isinstance(item, (list, tuple))]
        if not teachers: self.teacher_info_list = []; return
        t_cents_list = [item[1] for item in teachers]
        t_meta_list = [item[3] if len(item) == 4 else None for item in teachers]
        cfd_matrix = [[_cfd_distance(local_cents[lk].detach(), c_t[lk].detach()) for lk in local_cents] for c_t in
                      t_cents_list]
        cfd_scores = torch.tensor(cfd_matrix, device=self.device).mean(dim=1)
        normed_cfd = (cfd_scores - cfd_scores.min()) / (cfd_scores.max() - cfd_scores.min() + 1e-8)
        alpha_base = torch.softmax(-2.0 * normed_cfd, dim=0)
        layer_keys = list(local_cents.keys())
        T = len(teachers)
        maturity_mat = torch.ones(len(layer_keys), T, device=self.device)
        for t_idx, meta in enumerate(t_meta_list):
            if isinstance(meta, dict) and 'layer_maturity' in meta:
                for l_idx, lk in enumerate(layer_keys): maturity_mat[l_idx, t_idx] = max(
                    meta['layer_maturity'].get(lk, 1.0), self.maturity_eps)
        vmin = maturity_mat.min(dim=1, keepdim=True).values;
        vmax = maturity_mat.max(dim=1, keepdim=True).values
        maturity_norm = torch.clamp((maturity_mat - vmin) / (vmax - vmin + 1e-8), self.maturity_eps, 1.0)
        self.teacher_info_list = []
        for t in range(T):
            self.teacher_info_list.append({'centroids': t_cents_list[t], 'alpha': float(alpha_base[t].item()),
                                           'layer_maturity': {layer_keys[li]: float(maturity_norm[li, t].item()) for li
                                                              in range(len(layer_keys))}})

    def _compute_alignment_loss(self) -> torch.Tensor:
        """计算多教师 DKM 对齐损失。"""
        # (此部分内部逻辑不变，代码省略以保持简洁)
        if not self.teacher_info_list or not self.dkm_layers: return torch.zeros((), device=self.device)
        losses = []
        state = self.model.state_dict()
        for lk, dkm_layer in self.dkm_layers.items():
            W_flat = state[lk].view(-1, 1)
            t_cents = torch.stack([t['centroids'][lk].to(self.device) for t in self.teacher_info_list], dim=0)
            alpha_base = torch.tensor([t['alpha'] for t in self.teacher_info_list], device=self.device)
            maturity = torch.tensor([t['layer_maturity'][lk] for t in self.teacher_info_list], device=self.device)
            alpha_eff_raw = alpha_base * (maturity ** self.teacher_gamma)
            alpha_eff = F.normalize(alpha_eff_raw, p=1, dim=0)
            alpha_final = (1.0 - self.teacher_blend) * alpha_base + self.teacher_blend * alpha_eff
            alpha_final = F.normalize(alpha_final, p=1, dim=0)
            W_rec, _, _ = dkm_layer(W_flat, teacher_centroids=t_cents, teacher_alphas=alpha_final)
            losses.append(F.mse_loss(W_flat, W_rec))
        return torch.stack(losses).mean() if losses else torch.zeros((), device=self.device)

    def _cluster_and_prune_model_weights(self) -> Tuple[Dict, Dict, Dict]:
        """对模型权重进行聚类，并生成剪枝掩码 (质心为0则剪枝)。"""
        # (此部分内部逻辑不变，代码省略以保持简洁)
        clustered_state, cents, labels, mask = {}, {}, {}, {}
        state = self.model.state_dict()
        for key, w in state.items():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key:
                orig_shape = w.shape
                kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)
                flat_w = w.detach().view(-1, 1)
                kmeans.fit(flat_w)
                c_sorted, l_remapped = self._sort_centroids_and_remap(kmeans.centroids, kmeans.labels_)
                is_zero_centroid = (c_sorted.abs() < 1e-9).view(-1)
                mask[key] = ~is_zero_centroid[l_remapped].view(orig_shape)
                clustered_state[key] = c_sorted[l_remapped].view(orig_shape)
                cents[key] = c_sorted.to(self.device)
                labels[key] = l_remapped.view(-1).to(self.device)
            else:
                clustered_state[key] = w
                mask[key] = torch.ones_like(w, dtype=torch.bool)
        self.mask = mask
        self._update_local_history(cents, mask)
        return clustered_state, cents, labels

    # --- 其他辅助函数 (保持不变) ---
    def _prune_model_weights(self) -> Dict:
        pruned_dict = {}
        state = self.model.state_dict()
        for key, weight in state.items():
            pruned_dict[key] = weight * self.mask[key] if key in self.mask else weight
        return pruned_dict

    def _prepare_maturity_meta(self, cents: Dict, labels: Dict) -> Dict:
        layer_maturity = {}
        for lk, c_now in cents.items():
            m_now = self.mask[lk].to(c_now.device)
            l_now = labels[lk]
            maturity_vec = self._local_maturity(lk, c_now, l_now, m_now)
            layer_maturity[lk] = float(maturity_vec.mean().item())
        return {'layer_maturity': layer_maturity, 'sender_id': self.id, 'version': self.local_version,
                'sender_time': self.last_update_time}

    def _register_dkm_layers(self):
        self._train_fedavg()
        self.dkm_layers = {}
        for key in self.model.state_dict().keys():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key and 'conv' in key:
                self.dkm_layers[key] = MultiTeacherDKMLayer(n_clusters=self.n_clusters).to(self.device)

    @staticmethod
    def _sort_centroids_and_remap(c: torch.Tensor, lbl: torch.Tensor):
        order = torch.argsort(c.view(-1))
        sorted_c = c[order]
        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        return sorted_c, old2new[lbl]

    def _update_local_history(self, centroids: Dict, mask: Dict):
        for layer, c in centroids.items(): self._local_hist[layer].append(c.detach().cpu())
        for layer, m in mask.items(): self._local_mask_hist[layer].append(m.detach().cpu())

    def _local_maturity(self, lk: str, c: torch.Tensor, l: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if len(self._local_hist[lk]) < 2: return c.new_ones(c.shape[0], 1)
        stack = torch.stack(list(self._local_hist[lk]), dim=0)
        precision = 1.0 / (torch.var(stack, dim=0, unbiased=False) + self.maturity_eps)
        prev_c = self._local_hist[lk][-1].to(c.device)
        drift = torch.abs(c - prev_c)
        prev_m = self._local_mask_hist[lk][-1].to(m.device)
        flips = (prev_m ^ m).float().mean()
        stability = torch.exp(-self.beta_drift * drift - self.beta_mask * flips)
        return (precision * stability).clamp_min(1e-6)
