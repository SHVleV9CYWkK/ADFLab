from __future__ import annotations
from copy import deepcopy
from typing import Dict, Any, Tuple, List

import torch
from clients.client import Client


class SWIFTClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        self.comm_period: int = int(hyperparam.get("comm_period", 1))
        if self.comm_period <= 0:
            self.comm_period = 1

        self.comm_counter: int = 1
        self.ccs_weights: Dict[int, float] | None = hyperparam.get("ccs_weights", None)

    @torch.no_grad()
    def aggregate(self):
        if len(self.neighbor_model_weights_buffer) == 0:
            return

        current_state = {
            k: v.detach().clone().to(self.device)
            for k, v in self.model.state_dict().items()
        }

        neighbor_payloads: List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]] = \
            [(item[0], item[1]) for item in self.neighbor_model_weights_buffer]

        if self.ccs_weights is None:
            # 等权：自身 + 所有邻居同权
            num_neighbors = len(neighbor_payloads)
            total_nodes = num_neighbors + 1
            w_self = 1.0 / float(total_nodes)
            w_neighbor = 1.0 / float(total_nodes)

            # 初始化：先把自身权重乘上
            for k in current_state.keys():
                current_state[k] = current_state[k] * w_self

            # 累加邻居
            for state_dict, meta in neighbor_payloads:
                for k in current_state.keys():
                    current_state[k] += state_dict[k].to(self.device) * w_neighbor

        else:
            neighbor_ids = [meta["sender_id"] for _, meta in neighbor_payloads]
            raw_weights = [float(self.ccs_weights.get(cid, 0.0)) for cid in neighbor_ids]
            sum_neighbors = sum(raw_weights)

            if sum_neighbors >= 1.0:
                norm = sum_neighbors
                w_neighbors = [w / norm for w in raw_weights]
                w_self = 0.0
            else:
                w_neighbors = raw_weights
                w_self = 1.0 - sum_neighbors

            for k in current_state.keys():
                current_state[k] = current_state[k] * w_self

            for (state_dict, meta), w in zip(neighbor_payloads, w_neighbors):
                if w == 0.0:
                    continue
                for k in current_state.keys():
                    current_state[k] += state_dict[k].to(self.device) * w

        self.model.load_state_dict(current_state)

    def set_init_model(self, model: torch.nn.Module):
        self.model = deepcopy(model).to(self.device)

        if len(self.neighbor_model_weights_buffer) != 0:
            self.aggregate()

    def train(self):
        if (self.comm_counter % self.comm_period) == 0:
            # 只有在确实有邻居模型时才聚合
            if len(self.neighbor_model_weights_buffer) > 0:
                self.aggregate()
        self._local_train()

        self.comm_counter += 1

    # ---- 发送载荷 ----
    def send_model(self):
        return super().send_model()
