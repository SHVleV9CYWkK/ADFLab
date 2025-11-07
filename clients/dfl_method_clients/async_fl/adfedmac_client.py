# from __future__ import annotations
# from collections import deque
# from copy import deepcopy
# from typing import Any, Dict, List, Optional, Tuple
#
# import torch
#
# import math
#
# from clients.client import Client
# from utils.kmeans import TorchKMeans
#
#
# # --------------------------- 工具：CFD 距离 ---------------------------
# def _cfd_distance(centroids_a: torch.Tensor,
#                   centroids_b: torch.Tensor,
#                   n_freqs: int = 512,
#                   sigma: float = 1.0) -> float:
#     device = centroids_a.device
#     if centroids_a.ndim == 1:
#         centroids_a = centroids_a.view(-1, 1)
#         centroids_b = centroids_b.view(-1, 1)
#     D = centroids_a.shape[1]
#     freqs = torch.randn(n_freqs, D, device=device) * sigma
#     fa = (freqs @ centroids_a.T)
#     fb = (freqs @ centroids_b.T)
#     phi_a = torch.mean(torch.exp(1j * fa), dim=1)
#     phi_b = torch.mean(torch.exp(1j * fb), dim=1)
#     cfd = torch.mean(torch.abs(phi_a - phi_b) ** 2)
#     return cfd.item() if not isinstance(cfd, float) else cfd
#
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
#         super().__init__(client_id, dataset_index, full_dataset, hp, device)
#
#         self.neighbor_model_weights_buffer: List[Any] = []
#         self.cluster_model: Optional[
#             Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
#         ] = None
#         self.mask: Dict[str, torch.Tensor] = {}
#         self._local_hist: Dict[str, deque] = {}
#         self._local_mask_hist: Dict[str, deque] = {}
#         self._local_sem: Dict[str, torch.Tensor] = {}
#
#         # —— 必要超参 ——
#         self.n_clusters: int = int(hp.get('n_clusters', 16))
#         self.epochs: int = int(hp.get('epochs', 1))
#         self.base_decay_rate = hyperparam.get('base_decay_rate', 0.5)
#
#         # 相似度超参
#         # 注意: alpha_sem 和 beta_sem 不再使用
#         # beta_num 现在用作 Wasserstein 距离的温度系数
#         self.beta_num: float = float(hp.get('beta_num', 2.0))
#
#         # ... (成熟度, 新鲜度, 训练超参保持不变) ...
#         self.maturity_tau: float = float(hp.get('maturity_tau', 1e4))
#         self.maturity_eps: float = float(hp.get('maturity_eps', 1e-5))
#         self.beta_drift: float = float(hp.get('beta_drift', 2.0))
#         self.beta_invar: float = float(hp.get('beta_invar', 1.0))
#         self.beta_mask: float = float(hp.get('beta_mask', 2.0))
#         self.history_maxlen: int = int(hp.get('history_maxlen', 20))
#         self.lambda_time: float = float(hp.get('lambda_time', 0.05))
#         self.gamma_maturity: float = float(hp.get('gamma_maturity', 1.0))
#         self.sim_eps: float = 1e-8
#
#         self.ref_model = None
#         self.is_momentum = False
#
#     def _compute_ref_local_model_difference(self):
#         global_dict = self.ref_model.state_dict()
#         local_dict = self.model.state_dict()
#         difference_dict = {}
#         for key in global_dict:
#             difference_dict[key] = local_dict[key] - global_dict[key]
#         return difference_dict
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
#     def _ensure_hist_slot(self, layer: str):
#         if layer not in self._local_hist:
#             self._local_hist[layer] = deque(maxlen=self.history_maxlen)
#         if layer not in self._local_mask_hist:
#             self._local_mask_hist[layer] = deque(maxlen=self.history_maxlen)
#
#     def _update_local_history(self, centroids: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]):
#         for layer, c in centroids.items():
#             self._ensure_hist_slot(layer)
#             self._local_hist[layer].append(c.detach())
#         for layer, m in mask.items():
#             self._ensure_hist_slot(layer)
#             self._local_mask_hist[layer].append(m.detach())
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
#         last = hist[-1]
#         if len(hist) < 2:
#             return last.new_ones(last.shape[0], 1)
#
#         stack = torch.stack(list(hist), dim=0)  # [T,K,1]
#         mu = stack.mean(dim=0)  # [K,1]
#         var = ((stack - mu) ** 2).mean(dim=0)  # [K,1]
#
#         # 相对方差（对尺度鲁棒）
#         cv = var / (mu.abs() + 1e-4)  # [K,1]
#         lam = torch.exp(-self.maturity_tau * cv).clamp_min(self.maturity_eps)  # (eps,1]
#         return lam
#
#     def _stability_scores(self, layer_key: str,
#                           cents_now: torch.Tensor,
#                           labels_now: torch.Tensor,
#                           mask_now: torch.Tensor) -> torch.Tensor:
#         dev = cents_now.device
#         # 漂移：和上一帧质心差
#         if len(self._local_hist[layer_key]) >= 1:
#             prev_c = self._local_hist[layer_key][-1].to(dev)
#             drift = torch.abs(cents_now - prev_c)  # [K,1]
#         else:
#             drift = torch.zeros_like(cents_now)
#
#         # 簇内方差（当前帧；若担心开销，可对 flat_w 子采样）
#         flat_w = self.model.state_dict()[layer_key].to(dev).view(-1, 1).detach()
#         K = cents_now.shape[0]
#         invar = self._cluster_intra_var(flat_w, labels_now, K, eps=self.maturity_eps)  # [K,1]
#
#         # mask 翻转率（层级标量）
#         if len(self._local_mask_hist[layer_key]) >= 1:
#             prev_m = self._local_mask_hist[layer_key][-1].to(dev)
#             flips = (prev_m ^ mask_now).float().mean()  # scalar
#         else:
#             flips = torch.tensor(0.0, device=dev)
#
#         stab = torch.exp(
#             - self.beta_drift * drift
#             - self.beta_invar * invar
#             - self.beta_mask * flips
#         )
#         return stab
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
#         if not self.mask:
#             self.mask = {k: torch.ones_like(v) for k, v in self.model.state_dict().items()}
#         for layer_key, c_now in cents.items():
#             if c_now is None: continue
#             self._ensure_hist_slot(layer_key)
#             m_now = self.mask.get(layer_key)
#             l_now = labels.get(layer_key)
#             if m_now is None or l_now is None: continue
#             m_now = m_now.to(c_now.device)
#             maturity_vec = self._local_maturity(layer_key, c_now, l_now, m_now)
#             layer_maturity[layer_key] = float(maturity_vec.mean().item())
#
#         return {
#             'version_meta': 'dfedmac_meta_v1',
#             'sender_id': self.id,
#             'version': int(self.local_version),
#             'sender_time': float(self.last_update_time),
#             'layer_maturity': layer_maturity
#         }
#
#     def _local_train(self):
#         if self.client_train_loader is None:
#             raise RuntimeError("DataLoader not initialized; ensure init_client() on join.")
#
#         ref_momentum = 0.0
#         if self.is_momentum:
#             ref_momentum = self._compute_ref_local_model_difference()
#
#         self.model.train()
#
#         for _ in range(self.epochs):
#             for batch_idx, (x, labels) in enumerate(self.client_train_loader):
#                 self.model.load_state_dict(self._prune_model_weights())
#                 x, labels = x.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad(set_to_none=True)
#                 outputs = self.model(x)
#                 loss_final = self.criterion(outputs, labels).mean()
#                 loss_final.backward()
#
#                 if ref_momentum != 0.0:
#                     for name, param in self.model.named_parameters():
#                         if name in ref_momentum:
#                             param.grad += self.base_decay_rate * ref_momentum[name]
#
#                 self.optimizer.step()
#
#     @staticmethod
#     def _torch_1d_wasserstein(
#             C_a: torch.Tensor, C_b: torch.Tensor, a: torch.Tensor, b: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         【新】使用 PyTorch 计算两个 1D 离散分布之间的 Wasserstein-1 距离。
#         C_a, C_b: 质心值 (values) [K, 1]
#         a, b: 簇占比 (weights/probabilities) [K]
#
#         基于 W_1(F, G) = integral |CDF_F(x) - CDF_G(x)| dx
#         """
#         # 1. 确保输入正确
#         C_a = C_a.view(-1)
#         C_b = C_b.view(-1)
#
#         # 2. 对 A 排序
#         C_a_sorted, idx_a = torch.sort(C_a)
#         a_sorted = a[idx_a]
#
#         # 3. 对 B 排序
#         C_b_sorted, idx_b = torch.sort(C_b)
#         b_sorted = b[idx_b]
#
#         # 4. 获取所有值点，并排序去重
#         all_values = torch.cat([C_a_sorted, C_b_sorted])
#         all_values_sorted, _ = torch.sort(all_values)
#
#         # 5. 计算 deltas (dx)
#         # 我们计算点 x_i 和 x_i+1 之间的距离
#         deltas = all_values_sorted[1:] - all_values_sorted[:-1]  # [2K-1]
#
#         # 6. 计算 CDF 在 x_i 的值 (在 deltas 之前)
#         # cdf(x) = sum(w_i for v_i <= x)
#
#         # 6a. A 的 CDF
#         cdf_a_vals = torch.cumsum(a_sorted, dim=0)  # [K]
#         # 找到 all_values_sorted[:-1] 在 C_a_sorted 中的位置
#         indices_a = torch.searchsorted(C_a_sorted, all_values_sorted[:-1], right=True)  # [2K-1]
#         # cdf_a_at_x[j] = cdf_a_vals[indices_a[j]-1]
#         cdf_a_at_x = cdf_a_vals[torch.clamp(indices_a - 1, 0)]  # [2K-1]
#         cdf_a_at_x[indices_a == 0] = 0.0  # 处理边界
#
#         # 6b. B 的 CDF
#         cdf_b_vals = torch.cumsum(b_sorted, dim=0)  # [K]
#         indices_b = torch.searchsorted(C_b_sorted, all_values_sorted[:-1], right=True)
#         cdf_b_at_x = cdf_b_vals[torch.clamp(indices_b - 1, 0)]
#         cdf_b_at_x[indices_b == 0] = 0.0
#
#         # 7. 计算 W1 = sum |CDF_A(x_i) - CDF_B(x_i)| * dx_i
#         dist = torch.sum(torch.abs(cdf_a_at_x - cdf_b_at_x) * deltas)
#         return dist
#
#     @staticmethod
#     def _get_cluster_ratios(labels: Dict[str, torch.Tensor] | None,
#                             layer_key: str,
#                             K: int,
#                             device: torch.device
#                             ) -> torch.Tensor:
#         if (labels is not None) and (labels.get(layer_key) is not None):
#             l = labels[layer_key].to(device).view(-1)
#             counts = torch.bincount(l, minlength=K).float()
#         else:
#             counts = torch.ones(K, device=device)
#         return counts / (counts.sum() + 1e-8)
#
#
#     def _neighbor_similarity(self,
#                              local_cents: Dict[str, torch.Tensor],
#                              neighbor_cents: Dict[str, torch.Tensor],
#                              local_labels: Dict[str, torch.Tensor] | None,
#                              neighbor_labels: Dict[str, torch.Tensor] | None
#                              ) -> float:
#         """
#         【重写】
#         使用 1D Wasserstein (EMD) 距离计算相似度 S
#         """
#         per_d, per_sz = [], []  # 存储每层的距离(d)和规模(sz)
#
#         for layer_key, C in local_cents.items():
#             C_peer = neighbor_cents.get(layer_key, None)
#             if C is None or C_peer is None:
#                 continue
#
#             C = C.view(self.n_clusters, -1).float().to(self.device)
#             C_peer = C_peer.view(self.n_clusters, -1).float().to(self.device)
#             K = C.shape[0]
#
#             # 簇占比 a（本地），b（邻居）
#             a = self._get_cluster_ratios(local_labels, layer_key, K, self.device)
#             b = self._get_cluster_ratios(neighbor_labels, layer_key, K, self.device)
#
#             # 计算 EMD 距离
#             d_L = self._torch_1d_wasserstein(C, C_peer, a, b)
#
#             per_d.append(d_L)
#             per_sz.append(self._get_layer_size(layer_key))
#
#         if len(per_d) == 0:
#             return 0.5  # 默认
#
#         d_stack = torch.stack(per_d).to(self.device)
#         sz_stack = torch.tensor(per_sz, device=self.device, dtype=torch.float)
#         w_stack = sz_stack / (sz_stack.sum() + self.sim_eps)
#
#         # 归一化距离：使用所有层距离的中位数作为尺度
#         scale = d_stack.median() + self.sim_eps
#
#         # 将距离转换为相似度 (S)
#         # s_layers = exp(-beta * d_layer / scale)
#         s_layers = torch.exp(- self.beta_num * d_stack / scale)
#
#         # 按层规模加权平均相似度
#         s_scalar = float((s_layers * w_stack).sum().item())
#         return s_scalar
#
#     def _get_layer_size(self, layer_key: str) -> int:
#         m = self.mask.get(layer_key, None)
#         if m is not None:
#             return int(m.to(torch.int).sum().item())
#         if (self.cluster_model is not None) and (self.cluster_model[2].get(layer_key) is not None):
#             return int(self.cluster_model[2][layer_key].numel())
#         return int(self.model.state_dict()[layer_key].numel())
#
#
#     @torch.no_grad()
#     def _compute_peer_weights(self) -> List[float]:
#         """
#         【已更新】
#         调用新的 EMD _neighbor_similarity，不再返回 A 矩阵
#         """
#         if len(self.neighbor_model_weights_buffer) == 0:
#             return []
#
#         # 确保本地 cluster_model 存在
#         if self.cluster_model is None:
#             self.cluster_model = self._cluster_and_prune_model_weights()
#         _, local_cents, local_labels = self.cluster_model
#
#         t_now = float(getattr(self, "last_update_time", 0.0))
#         scores_raw = []
#
#         for (peer_state, peer_cents, peer_labels, peer_meta) in self.neighbor_model_weights_buffer:
#
#             # 1. 相似度 (S) - 使用 EMD
#             s = self._neighbor_similarity(
#                 local_cents=local_cents,
#                 neighbor_cents=peer_cents,
#                 local_labels=local_labels,
#                 neighbor_labels=peer_labels
#             )
#
#             # 2. 成熟度 (M)
#             m = 1.0  # 默认
#             if isinstance(peer_meta, dict) and ('layer_maturity' in peer_meta):
#                 ms, szs = [], []
#                 for layer_key, c_local in local_cents.items():
#                     ml = peer_meta['layer_maturity'].get(layer_key, None)
#                     if ml is None: continue
#                     ms.append(float(ml))
#                     szs.append(self._get_layer_size(layer_key))
#
#                 if ms:
#                     ms_t = torch.tensor(ms, device=self.device).float()
#                     sz_t = torch.tensor(szs, device=self.device).float()
#                     m = float((ms_t * (sz_t / (sz_t.sum() + 1e-8))).sum().item())
#
#             m_score = max(self.sim_eps, m) ** self.gamma_maturity
#
#             # 3. 新鲜度 (R)
#             sender_time = float(peer_meta.get('sender_time', t_now)) if isinstance(peer_meta, dict) else t_now
#             r = math.exp(- self.lambda_time * max(0.0, t_now - sender_time))
#
#             # 最终得分
#             score = max(self.sim_eps, s) * m_score * r
#             scores_raw.append(score)
#
#         Z = sum(scores_raw) + self.sim_eps
#         rel_weights = [s / Z for s in scores_raw]
#         return rel_weights
#
#     @torch.no_grad()
#     def aggregate(self):
#         """
#         【已恢复】
#         恢复到您原始的 "普通平均" (W-Space) 聚合。
#         此方法在数学上是稳健的，它聚合的是 "近似模型" 的值。
#         """
#         n = len(self.neighbor_model_weights_buffer)
#         if n == 0:
#             self.is_momentum = False
#             return  # 无邻居或无训练就不聚合
#
#         if self.cluster_model is None:
#             neighbor_states = [tpl[0] for tpl in self.neighbor_model_weights_buffer]
#             if len(neighbor_states) == 0:
#                 return
#             avg_state = {}
#             keys = set(neighbor_states[0].keys())
#             for k in keys:
#                 acc = torch.stack([st[k].to(self.device) for st in neighbor_states]).mean(dim=0)
#                 avg_state[k] = acc
#             self.model.load_state_dict(avg_state)
#             self.cluster_model = self._cluster_and_prune_model_weights()
#             self.is_momentum = False
#             return
#
#         # 1) 本地最新质心（用于计算S）
#         _, local_cents, _ = self.cluster_model
#
#         # 2) 邻居相对权重（和为 1）
#         neighbor_rel = self._compute_peer_weights()  # 长度 n，∑=1
#         if len(neighbor_rel) != n:
#             self.is_momentum = False
#             return
#
#         neighbor_states = [tpl[0] for tpl in self.neighbor_model_weights_buffer]
#         keys = set(neighbor_states[0].keys())
#
#
#         # 4) 计算加权均值（邻居 + 本地）
#         avg = {}
#
#         # 先邻居
#         for k in keys:
#             acc = None
#             for st, w in zip(neighbor_states, neighbor_rel):
#                 if k not in st: continue  # 安全检查
#                 t = st[k].to(self.device)
#                 acc = t.mul(w) if acc is None else acc.add(t, alpha=w)
#             avg[k] = acc
#
#         self.ref_model.load_state_dict(avg)
#         self.is_momentum = True
#
#     def train(self):
#         """
#         - 普通客户端：训练前可调用 aggregate()（由上层决定），训练本体走 _local_train()。
#         - 延迟客户端：
#             * 预训练阶段：与普通客户端一致（可聚合 + _local_train()）；
#             * 预训练结束：切换对齐模式（监督 + 对齐损失）。
#         """
#         self._local_train()
#         self.cluster_model = self._cluster_and_prune_model_weights()
#
#     def set_init_model(self, model: torch.nn.Module):
#         self.model = deepcopy(model).to(self.device)
#         self.ref_model = deepcopy(model)
#
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
#
#         return clustered_state_dict, cents, labels, meta
