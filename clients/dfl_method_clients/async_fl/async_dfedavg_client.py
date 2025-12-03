from __future__ import annotations
from copy import deepcopy
from typing import Dict

import torch
from clients.client import Client


class AsyncDFedAvgClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

    # ---- 聚合：平均(自己 + 邻居) ----
    @torch.no_grad()
    def aggregate(self):
        if len(self.neighbor_model_weights_buffer) == 0:
            return  # 没有邻居更新就不动

        device = self.device
        neighbor_states = [neighbor[0] for neighbor in self.neighbor_model_weights_buffer]

        # 本地 state
        local_state = self.model.state_dict()
        count = 1 + len(neighbor_states)

        new_state = {}

        for k, v_local in local_state.items():
            t0 = v_local.to(device)

            if t0.is_floating_point() or t0.is_complex():
                acc = t0.clone()
                for sd in neighbor_states:
                    acc.add_(sd[k].to(device))
                acc.div_(float(count))
                new_state[k] = acc

            else:
                new_state[k] = t0

        self.model.load_state_dict(new_state)
        self.neighbor_model_weights_buffer.clear()

    # ---- 初始化模型 ----
    def set_init_model(self, model):
        """
        Join 时设置初始模型。若此刻缓冲里已经有邻居更新（一般很少见），则做一次聚合以充分利用信息。
        """
        self.model = deepcopy(model)
        if len(self.neighbor_model_weights_buffer) != 0:
            self.aggregate()

    # ---- 本地训练 ----
    def train(self):
        self._local_train()

    # ---- 发送载荷 ----
    def send_model(self) -> Dict:
        """
        返回可传输载荷（CPU state + meta）。直接复用基类实现，包含 version / sender_time / nbytes 等信息。
        """
        return super().send_model()