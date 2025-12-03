from __future__ import annotations

from copy import deepcopy
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from clients.client import Client


class DivShareClient(Client):
    """
    多 fragment 版的 DivShare 风格客户端：
    - send_model(): 把当前模型切成多个 fragments，一次性打包发送；
    - receive_neighbor_model(): 复用基类逻辑，把 (fragments_list, meta) 存进 buffer；
    - aggregate(): 对每个邻居随机挑一个 fragment，只用这个 fragment 里的参数做平均。
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        # 碎片相关超参数
        # 每次 train 后，希望切成多少块 fragment（论文里大概是 1/Ω 块，你可以直接传进来）
        self.num_fragments: int = int(hyperparam.get("num_fragments", 10))
        self.num_fragments = max(1, self.num_fragments)

        # 至少让每个 fragment 里有若干个参数 tensor，避免太稀
        self.min_params_per_fragment: int = int(hyperparam.get("min_params_per_fragment", 1))

        # 方便缓存参数名
        self._param_keys_cache: List[str] | None = None

    # ------------------------------------------------------------------
    # 初始化模型
    # ------------------------------------------------------------------
    def set_init_model(self, model: torch.nn.Module):
        self.model = deepcopy(model)
        # 如果已经有缓冲的邻居更新，可以立刻聚合利用
        if len(self.neighbor_model_weights_buffer) != 0:
            self.aggregate()

    # ------------------------------------------------------------------
    # 本地训练（直接复用基类）
    # ------------------------------------------------------------------
    def train(self):
        self._local_train()
        # 训练完正常走 on_train_done()（由协调器调用），这里不用管时间/计数

    # ------------------------------------------------------------------
    # 工具：把当前模型切成多块 fragments
    # ------------------------------------------------------------------
    def _make_fragments_from_current_model(self) -> List[Dict[str, torch.Tensor]]:
        if self.model is None:
            raise RuntimeError("模型尚未设置，无法切 fragment")

        # 完整 state_dict 放在 CPU 上
        full_state: Dict[str, torch.Tensor] = {
            k: v.detach().clone().cpu()
            for k, v in self.model.state_dict().items()
        }

        # 初始化 / 复用参数名列表
        if self._param_keys_cache is None:
            self._param_keys_cache = list(full_state.keys())
        keys = self._param_keys_cache.copy()
        n_keys = len(keys)

        # 确定真正要用的 fragment 数（不能比参数 tensor 数多）
        num_frag = min(self.num_fragments, n_keys)
        if num_frag <= 0:
            num_frag = 1

        # 打乱参数名顺序，然后切成 num_frag 份
        np.random.shuffle(keys)
        chunks = np.array_split(keys, num_frag)

        fragments: List[Dict[str, torch.Tensor]] = []
        for chunk_keys in chunks:
            ck = list(chunk_keys)
            if len(ck) == 0:
                continue
            # 保证每个 fragment 至少有 min_params_per_fragment 个 tensor：
            # 如果太少，就和下一个合并（简单处理）
            if len(ck) < self.min_params_per_fragment and fragments:
                # 附加到上一个 fragment
                last_frag = fragments[-1]
                for k in ck:
                    last_frag[k] = full_state[k]
                continue

            frag_state: Dict[str, torch.Tensor] = {k: full_state[k] for k in ck}
            fragments.append(frag_state)

        if not fragments:
            # 极端情况兜底：整个模型作为一个 fragment
            fragments = [full_state]

        return fragments

    # ------------------------------------------------------------------
    # 发送：一次性发送多块 fragment，由接收端自行选择
    # ------------------------------------------------------------------
    def send_model(self) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
        """
        返回 (fragments_list, meta)：
        - fragments_list: List[Dict[str, Tensor]]，每个 dict 是一个 fragment；
        - meta: 包含 sender_id / version / sender_time 等。
        协调器仍然按照原逻辑，把整个 payload 发给若干邻居。
        """
        if self.model is None:
            raise RuntimeError("send_model() 之前必须先设置模型：set_init_model(model)")

        fragments = self._make_fragments_from_current_model()

        meta: Dict[str, Any] = {
            "sender_id": self.id,
            "version": self.local_version,
            "sender_time": self.last_update_time,
            "is_multifragment": True,
            "num_fragments": len(fragments),
        }

        # payload 仍然是一个二元组，兼容基类 receive_neighbor_model 的逻辑
        return fragments, meta

    # ------------------------------------------------------------------
    # 聚合：对每个邻居随机选一个 fragment，只融合那一块
    # ------------------------------------------------------------------
    @torch.no_grad()
    def aggregate(self):
        """
        从 neighbor_model_weights_buffer 里取出多个 (fragments_list, meta)，
        对每个邻居随机挑一个 fragment，只用该 fragment 里的参数与本地模型做平均。

        规则：
        - 对参数名 k：
          - 若有邻居选中的 fragment 包含 k：
                new_k = (local_k + sum(neighbor_k)) / (1 + count_neighbors_for_k)
          - 否则：保留 local_k 不变。
        """
        if len(self.neighbor_model_weights_buffer) == 0:
            return

        if self.model is None:
            raise RuntimeError("aggregate() 之前必须先设置模型")

        device = self.device

        # 本地模型参数（device 上）
        local_state: Dict[str, torch.Tensor] = {
            k: v.detach().clone().to(device)
            for k, v in self.model.state_dict().items()
        }

        neighbor_payloads = self.neighbor_model_weights_buffer
        # 用完清空缓冲
        self.neighbor_model_weights_buffer = []

        # 收集：参数名 -> 若干邻居给出的该参数 tensor
        neighbor_param_buckets: Dict[str, List[torch.Tensor]] = {}

        for payload in neighbor_payloads:
            # payload: (fragments_list, meta)
            if not isinstance(payload, (list, tuple)) or len(payload) != 2:
                continue
            fragments_list, meta = payload

            if not isinstance(fragments_list, list) or len(fragments_list) == 0:
                continue

            idx = np.random.randint(len(fragments_list))
            chosen_frag = fragments_list[idx]
            if not isinstance(chosen_frag, dict):
                continue

            for k, tensor in chosen_frag.items():
                if k not in local_state:
                    continue
                if k not in neighbor_param_buckets:
                    neighbor_param_buckets[k] = []
                neighbor_param_buckets[k].append(tensor.to(device))

        # 逐参数聚合
        new_state: Dict[str, torch.Tensor] = {}
        for k, local_tensor in local_state.items():
            if k not in neighbor_param_buckets:
                new_state[k] = local_tensor
            else:
                bucket = neighbor_param_buckets[k]
                t0 = local_tensor

                if t0.is_floating_point() or t0.is_complex():
                    acc = t0.clone()
                    for t in bucket:
                        acc.add_(t.to(device, dtype=t0.dtype))
                    acc.div_(float(1 + len(bucket)))
                    new_state[k] = acc
                else:
                    # 非浮点：保留本地值（或改成 bucket[0] 也行）
                    new_state[k] = t0

        self.model.load_state_dict(new_state)
