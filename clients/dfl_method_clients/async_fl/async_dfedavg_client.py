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
        if len(self.neighbor_model_weights) == 0:
            return  # 没有邻居更新就不动

        # 取当前本地模型（作为被平均的第一项）
        current = {k: v.detach().clone().to(self.device) for k, v in self.model.state_dict().items()}
        count = 1 + len(self.neighbor_model_weights)

        # 累加邻居
        for sd in self.neighbor_model_weights:
            for k in current.keys():
                current[k] += sd[k].to(self.device)

        # 做平均并加载回模型
        for k in current.keys():
            current[k] /= float(count)

        self.model.load_state_dict(current)

    # ---- 初始化模型 ----
    def set_init_model(self, model):
        """
        Join 时设置初始模型。若此刻缓冲里已经有邻居更新（一般很少见），则做一次聚合以充分利用信息。
        """
        self.model = deepcopy(model)
        if len(self.neighbor_model_weights) != 0:
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