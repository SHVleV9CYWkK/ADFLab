from __future__ import annotations
from copy import deepcopy
from typing import Dict

import torch
from clients.client import Client


class AsyncDFedAvgClient(Client):
    """
    异步去中心化 FedAvg：
    - 聚合策略：简单平均（等权），但 **包含本地模型** 与邻居缓冲中的模型。
      这样在异步“收到即融合”的设置下，不会被单个邻居的模型直接覆盖。
    - 训练：沿用基类 _local_train()
    - 发送：复用基类 send_model()（返回 CPU state_dict + meta）
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

    # ---- 聚合：平均(自己 + 邻居) ----
    @torch.no_grad()
    def aggregate(self):
        if len(self.neighbor_model_weights_buffer) == 0:
            return  # 没有邻居更新就不动

        n = len(self.neighbor_model_weights_buffer)
        neighbor_rel = [1.0 / n] * n
        neighbor_states = [tpl[0] for tpl in self.neighbor_model_weights_buffer]
        keys = set(neighbor_states[0].keys())

        local_w = 1.0 / (n + 1.0)
        scale_neighbors = 1.0 - local_w
        neighbor_w = [w * scale_neighbors for w in neighbor_rel]

        # 4) 计算加权均值（邻居 + 本地）
        avg = {}
        local_state_dict = self.model.state_dict()

        # 先邻居
        for k in keys:
            acc = None
            for st, w in zip(neighbor_states, neighbor_w):
                if k not in st: continue  # 安全检查
                t = st[k].to(self.device)
                acc = t.mul(w) if acc is None else acc.add(t, alpha=w)

            if k in local_state_dict and acc is not None:
                t_local = local_state_dict[k].to(self.device)
                acc = acc.add(t_local, alpha=local_w)
                avg[k] = acc
            elif k in local_state_dict:  # 仅有本地
                avg[k] = local_state_dict[k].to(self.device)

        self.model.load_state_dict(avg)

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