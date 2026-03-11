from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Any

import torch
from clients.client import Client


class ADFedPushSumClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        self._join_anchor_state = None
        self.is_delayed_client = hyperparam['is_delayed']
        if self.is_delayed_client:
            self.ps_mass: float = 0.0
        else:
            self.ps_mass: float = 1.0

    def _prepare_maturity_meta(self) -> Dict[str, Any]:
        d_out = self.k_push + 1
        w_before = float(self.ps_mass)
        share = w_before / float(d_out)
        self.ps_mass = share

        return {
            "version_meta": "cadfedfilter_meta_v1",
            "sender_id": self.id,
            "version": self.local_version,
            "sender_time": self.last_update_time,
            "ps_mass_share": share,
        }


    @torch.no_grad()
    def aggregate(self):
        if not self.neighbor_model_weights_buffer:
            return

        device = self.device

        # 1) 先收集邻居 state 和 mass
        neighbor_states: List[Dict[str, torch.Tensor]] = []
        neighbor_masses: List[float] = []

        for tpl in self.neighbor_model_weights_buffer:
            st = tpl[0]
            neighbor_states.append(st)

            meta = tpl[-1] if isinstance(tpl[-1], dict) else {}
            neighbor_masses.append(float(meta.get("ps_mass_share", 0.0)))

        if not neighbor_states:
            self.neighbor_model_weights_buffer.clear()
            return

        # 2) 固定聚合前本地质量
        w_old = float(self.ps_mass)
        w_in = float(sum(neighbor_masses))
        w_new = w_old + w_in

        masses_tensor = torch.tensor(neighbor_masses, device=device, dtype=torch.float32)

        # 3) 用 w_old / neighbor_masses 做加权平均
        local_state = self.model.state_dict()
        avg_state: Dict[str, torch.Tensor] = {}

        for k, v_local in local_state.items():
            t_local = v_local.to(device)

            if "bn" in k:
                avg_state[k] = t_local
                continue

            nb_tensors: List[torch.Tensor] = []
            valid_indices: List[int] = []
            for i, st in enumerate(neighbor_states):
                if k in st:
                    nb_tensors.append(st[k].to(device))
                    valid_indices.append(i)

            if not nb_tensors:
                avg_state[k] = t_local
                continue

            if t_local.is_floating_point() or t_local.is_complex():
                nb_stack = torch.stack(nb_tensors, dim=0)
                idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
                w_subset = masses_tensor.index_select(0, idx_tensor)

                view_shape = [len(nb_tensors)] + [1] * (nb_stack.ndim - 1)
                w_view = w_subset.view(*view_shape)

                weighted_sum = (nb_stack * w_view).sum(dim=0) + t_local * w_old
                avg_state[k] = weighted_sum / w_new
            else:
                avg_state[k] = t_local

        # 4) 更新模型与质量
        self.model.load_state_dict(avg_state)
        self.ps_mass = w_new

        self.neighbor_model_weights_buffer.clear()

    # ---- 初始化模型 ----
    def set_init_model(self, model):
        """
        Join 时设置初始模型。若此刻缓冲里已经有邻居更新（一般很少见），则做一次聚合以充分利用信息。
        """
        self.model = deepcopy(model)
        # 保存 delayed join 的 anchor（W0）
        self._join_anchor_state = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        if len(self.neighbor_model_weights_buffer) != 0:
            self.aggregate()

    # ---- 本地训练 ----
    def train(self):
        self._local_train()

    # ---- 发送载荷 ----
    def send_model(self):
        if self.model is None:
            raise RuntimeError("send_model() 之前必须先设置模型：set_init_model(model)")
        state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        meta = self._prepare_maturity_meta()
        return state, meta

    def receive_neighbor_model(self, neighbor_model):
        """
        Push-Sum 安全接收：
        - 不允许因 buffer_limit / replace 丢质量
        - 只能合并消息
        """
        state, meta = neighbor_model
        sender_id = meta.get("sender_id", None)
        w_new = float(meta.get("ps_mass_share", 0.0))

        # 1) 如果启用 replace_same_sender：不要 pop 丢掉，而是“合并质量”
        if self.is_replace_same_client_model and sender_id is not None:
            for idx, item in enumerate(self.neighbor_model_weights_buffer):
                st_old, meta_old = item
                if meta_old.get("sender_id", None) == sender_id:
                    w_old = float(meta_old.get("ps_mass_share", 0.0))
                    w_sum = w_old + w_new
                    if w_sum <= 1e-12:
                        # 两个都几乎为 0，保留新的即可
                        self.neighbor_model_weights_buffer[idx] = (state, meta)
                        break

                    # 合并 state：theta = (w_old*theta_old + w_new*theta_new) / w_sum
                    merged_state = {}
                    for k, v_new in state.items():
                        v_old = st_old.get(k, None)
                        if v_old is None:
                            merged_state[k] = v_new
                            continue
                        # 只对浮点/复数加权；非浮点保持新的
                        if torch.is_tensor(v_new) and (v_new.is_floating_point() or v_new.is_complex()):
                            merged_state[k] = (v_old * w_old + v_new * w_new) / w_sum
                        else:
                            merged_state[k] = v_new

                    merged_meta = dict(meta)
                    merged_meta["ps_mass_share"] = w_sum
                    self.neighbor_model_weights_buffer[idx] = (merged_state, merged_meta)
                    return  # 已合并完成，直接返回

        # 2) 常规追加
        self.neighbor_model_weights_buffer.append((state, meta))

        # 3) buffer_limit 溢出：不要丢，改为“合并最旧的 overflow 条”
        if self.buffer_limit is not None and self.buffer_limit > 0:
            overflow = len(self.neighbor_model_weights_buffer) - self.buffer_limit
            if overflow > 0:
                # 取最旧的 overflow 条
                to_merge = self.neighbor_model_weights_buffer[:overflow]
                keep = self.neighbor_model_weights_buffer[overflow:]

                # 合并这 overflow 条
                w_sum = 0.0
                # 用第一条的 keys 作为基准（也可做 union）
                merged_state = None

                for st_i, meta_i in to_merge:
                    w_i = float(meta_i.get("ps_mass_share", 0.0))
                    if w_i <= 0.0:
                        continue
                    w_sum += w_i
                    if merged_state is None:
                        # 初始化：先复制一份加权的
                        merged_state = {}
                        for k, v in st_i.items():
                            if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex()):
                                merged_state[k] = v * w_i
                            else:
                                # 非浮点直接取（或忽略）
                                merged_state[k] = v
                    else:
                        for k, v in st_i.items():
                            if k not in merged_state:
                                if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex()):
                                    merged_state[k] = v * w_i
                                else:
                                    merged_state[k] = v
                            else:
                                if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex()):
                                    merged_state[k] = merged_state[k] + v * w_i
                                # 非浮点不累加

                if merged_state is None or w_sum <= 1e-12:
                    # 没法合并（全是 0 质量），那就保留 keep（不追加合并项）
                    self.neighbor_model_weights_buffer = keep
                    return

                # 从 “加权和” 变回 “加权平均”
                for k, v in merged_state.items():
                    if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex()):
                        merged_state[k] = v / w_sum

                merged_meta = {
                    "sender_id": -1,  # 汇总项，虚拟 sender
                    "ps_mass_share": w_sum,
                }

                # 保持 buffer 长度不超过 limit：用 1 条汇总项 + keep
                self.neighbor_model_weights_buffer = [(merged_state, merged_meta)] + keep
