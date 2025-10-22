from __future__ import annotations

import heapq
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

@dataclass(order=True)
class _Event:
    # heapq 以元组顺序比较，这里把排序键放前面
    due: float
    prio: int
    seq: int
    kind: str           # 'JOIN' | 'TRAIN_DONE' | 'EVAL_TICK'
    cid: Optional[int]  # 与客户端相关的事件则为客户端 id
    epoch: int          # 懒惰失效：事件创建时的 epoch 快照


class AsyncCoordinator:
    """
    异步去中心化协调器（无 Rewire、无 Join 预热、保留在线过滤 + 少推）。

    参数
    ----
    clients: List[Client]                客户端对象列表（索引 == client.id）
    model: torch.nn.Module               初始模型（会 deepcopy 分发）
    device: torch.device                 当前设备（未直接使用，仅保持接口一致）
    client_delay_dict: Dict[int, float]  加入时间表：client_id -> join_time（单位=模拟时间）；缺省视为 0.0
    args: 命名空间/对象，需含：
        - num_conn: int          覆盖图出度目标（无向时是度；有向时是出度）
        - symmetry: int          非 0 表示无向覆盖图；0 表示有向覆盖图
        - gossip: int            和旧代码兼容，不使用；保留字段
        - seed: int              全局随机种子
        - k_push: Optional[int]  每次训练完成要推送的邻居数（≤ num_conn），缺省为 num_conn
        - eval_interval: float   评估间隔（秒/任意时间单位）；<=0 则不自动评估
    终止条件由 run(...) 传入（until_time / max_events）。
    """

    # 事件优先级（同刻处理顺序）：JOIN -> TRAIN_DONE -> EVAL_TICK
    _PRIO = {"JOIN": 1, "TRAIN_DONE": 2, "EVAL_TICK": 3}

    def __init__(self, clients, model, device, client_delay_dict: Dict[int, float], args):
        # ---- 基本状态 ----
        self.all_clients = clients
        self.num_clients = len(self.all_clients)
        self.init_model = model
        self.device = device

        # Join 时间表：缺省 0.0
        self.join_time: Dict[int, float] = {i: 0.0 for i in range(self.num_clients)}
        if client_delay_dict:
            for cid, t in client_delay_dict.items():
                self.join_time[int(cid)] = float(t)

        # 覆盖图超参
        self.num_conn = int(getattr(args, "num_conn", 2))
        self.symmetry = int(getattr(args, "symmetry", 1))
        self.gossip = int(getattr(args, "gossip", 0))  # 保留兼容，未使用

        # 推送与评估
        self.k_push = int(getattr(args, "k_push", self.num_conn))
        self.eval_interval = float(getattr(args, "eval_interval", 0.0))  # <=0 不评估

        # 随机性（可复现）
        self.seed = int(getattr(args, "seed", 0))
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

        # 在线/epoch（懒惰失效）
        self.online: List[bool] = [False] * self.num_clients
        self.epoch: List[int] = [0] * self.num_clients

        # 事件堆
        self._heap: List[_Event] = []
        self._seq_counter: int = 0
        self.now: float = 0.0

        # 初始化：调度 JOIN / TRAIN_DONE / EVAL_TICK
        self._bootstrap_events()

    # ----------------------------------------------------------------------
    # 初始化事件：按 join_time 投递 JOIN / TRAIN_DONE；投 EVAL_TICK
    # ----------------------------------------------------------------------
    def _bootstrap_events(self):
        # 初始在线与延迟加入
        for cid in range(self.num_clients):
            jt = float(self.join_time.get(cid, 0.0))
            if jt <= 0.0:
                # 立即上线
                self._do_join(cid, at_time=0.0, schedule_only=True)
            else:
                # 以后再上线
                self._push_event(jt, "JOIN", cid, self.epoch[cid])

        # 周期评估
        if self.eval_interval > 0:
            self._push_event(self.eval_interval, "EVAL_TICK", None, 0)

    # ----------------------------------------------------------------------
    # 事件循环
    # ----------------------------------------------------------------------
    def run(self, until_time: Optional[float] = None, max_events: Optional[int] = None):
        """
        执行事件循环：
        - until_time: 运行到给定模拟时间（含）即停止；None 表示不设时间上限
        - max_events: 处理给定数量的事件后停止；None 表示不设事件数上限
        """
        processed = 0
        while self._heap:
            ev = heapq.heappop(self._heap)
            # 终止条件（时间）
            if until_time is not None and ev.due > until_time:
                heapq.heappush(self._heap, ev)
                break

            # 推进时间
            self.now = ev.due

            # 懒惰失效：过期事件跳过
            if ev.cid is not None and ev.epoch != self.epoch[ev.cid]:
                continue

            if ev.kind == "JOIN":
                self._handle_join(ev.cid)
            elif ev.kind == "TRAIN_DONE":
                self._handle_train_done(ev.cid)
            elif ev.kind == "EVAL_TICK":
                self._handle_eval_tick()
            else:
                raise RuntimeError(f"Unknown event kind: {ev.kind}")

            processed += 1
            if max_events is not None and processed >= max_events:
                break

    # ----------------------------------------------------------------------
    # 事件处理器
    # ----------------------------------------------------------------------
    def _handle_join(self, cid: int):
        """Join 发生：初始化客户端，调度首个 TRAIN_DONE（无预热）。"""
        self._do_join(cid, at_time=self.now, schedule_only=False)

    def _do_join(self, cid: int, at_time: float, schedule_only: bool):
        if not self.online[cid]:
            # 初始化模型与资源
            self.online[cid] = True
            self.epoch[cid] += 1  # invalidate 既往事件（若有）
            self.all_clients[cid].set_init_model(deepcopy(self.init_model))
            self.all_clients[cid].init_client()

        # 无预热：直接排程首个 TRAIN_DONE
        dur = float(self.all_clients[cid].compute_time_for_next_burst())
        self._push_event(at_time + dur, "TRAIN_DONE", cid, self.epoch[cid])

    def _handle_train_done(self, cid: int):
        """
        本地训练完成：更新元信息 -> （必要时先聚合本地缓冲）-> 发送 -> 排程下一次。
        """
        client = self.all_clients[cid]

        buf = getattr(client, "neighbor_model_weights", None)
        if buf is not None and len(buf) > 0:
            client.aggregate()

        # 1) 本地训练（子类可在 train() 内调用 _local_train()）
        client.train()

        # 2) 更新异步元信息
        client.on_train_done(self.now)

        # 4) 在线过滤 + 少推：选择目标邻居
        receivers = self._sample_online_neighbors(cid, self.k_push)

        # 5) 外发（即时到达）
        payload = client.send_model()
        for r in receivers:
            self.all_clients[r].receive_neighbor_model(payload)

        # 6) 排程下一次 TRAIN_DONE
        dur = float(client.compute_time_for_next_burst())
        self._push_event(self.now + dur, "TRAIN_DONE", cid, self.epoch[cid])

    def _handle_eval_tick(self):
        """时间对齐评估：仅统计 online 客户端；随后调度下一次评估并触发 LR 调度。"""
        if self.eval_interval <= 0:
            return

        # 评估
        online_ids = [i for i, on in enumerate(self.online) if on]
        if len(online_ids) > 0:
            client_results: Dict[int, Dict[str, float]] = {}
            # 计算宏平均
            agg: Dict[str, float] = {}
            for cid in online_ids:
                metrics = self.all_clients[cid].evaluate_model()
                client_results[cid] = metrics
                for k, v in metrics.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
            overall = {k: (v / len(online_ids)) for k, v in agg.items()}

            # 可在此处输出/记录 overall 与 client_results（留给上层调用者）

            # 学习率调度（基于 last_accuracy）
            for cid in online_ids:
                self.all_clients[cid].update_lr()

        # 调度下一次 EVAL_TICK\
        print(f"EVAL_TICK self.now: {self.now}, self.eval_interval:{self.eval_interval}， Total:{self.now + self.eval_interval}")
        self._push_event(self.now + self.eval_interval, "EVAL_TICK", None, 0)

    # ----------------------------------------------------------------------
    # 邻居采样（在线过滤 + 少推）
    # ----------------------------------------------------------------------
    def _sample_online_neighbors(self, cid: int, k: int) -> List[int]:
        """
        在“出邻居 ∩ 已上线”中均匀无放回采样至多 k 个；候选不足则全部返回；没有候选则空。
        """
        # 在线候选（全连通，不看 connected_graph）
        candidates = [j for j in range(self.num_clients) if j != cid and self.online[j]]
        n = len(candidates)
        if n == 0:
            return []
        if k >= n:
            # 少推：不补位
            return candidates[:]  # 返回一个副本
        # 使用事先构造好的可复现实验用 RNG，例如 self._rng = random.Random(self.seed)
        return self._rng.sample(candidates, k)
    # ----------------------------------------------------------------------
    # 事件工具
    # ----------------------------------------------------------------------
    def _push_event(self, due: float, kind: str, cid: Optional[int], epoch: int):
        ev = _Event(
            due=float(due),
            prio=self._PRIO[kind],
            seq=self._next_seq(),
            kind=kind,
            cid=cid,
            epoch=int(epoch),
        )
        heapq.heappush(self._heap, ev)

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter