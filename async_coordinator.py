from __future__ import annotations

import heapq
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, Future
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
    异步去中心化协调器（可串行/并行两种模式）：

    - max_workers == -1：串行事件循环（与原实现一致）。
    - max_workers >= 1：多线程并发训练（TRAIN_DONE 提交线程池；EVAL_TICK 为屏障，等待在途训练完成后统一评估）。

    参数
    ----
    clients: List[Client]
    model: torch.nn.Module
    device: torch.device
    client_delay_dict: Dict[int, float]   # client_id -> join_time；缺省视为 0.0
    args: 需含
        - num_conn: int
        - symmetry: int
        - gossip: int                     # 未使用，保留
        - seed: int
        - k_push: Optional[int]
        - eval_interval: float            # <=0 则不评估
    max_workers: int
        - -1  => 串行
        - >=1 => 并行线程数
    """

    # 事件优先级（同刻处理顺序）：JOIN -> TRAIN_DONE -> EVAL_TICK
    _PRIO = {"JOIN": 1, "TRAIN_DONE": 2, "EVAL_TICK": 3}

    def __init__(self, clients, model, device, client_delay_dict: Dict[int, float], args,
                 max_workers: int = -1):
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

        # ---- 并行控制 ----
        self._serial = (max_workers == -1)
        self._lock = threading.RLock()  # 仅并行模式下需要强一致，但串行也可以复用
        self._heap_lock = self._lock

        self._executor: Optional[ThreadPoolExecutor] = None
        self._inflight: Dict[int, Future] = {}  # cid -> Future（在途训练）
        self._eval_barrier: bool = False        # 评估屏障：开启后阻止新一轮训练排程

        if not self._serial:
            if max_workers is None or max_workers < 1:
                # 默认给一个合理值
                max_workers = min(self.num_clients or 1, (os.cpu_count() or 4))
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # 初始化：调度 JOIN / TRAIN_DONE / EVAL_TICK
        self._bootstrap_events()

    # ----------------------------------------------------------------------
    # 初始化事件：按 join_time 投递 JOIN / TRAIN_DONE；投 EVAL_TICK
    # ----------------------------------------------------------------------
    def _bootstrap_events(self):
        for cid in range(self.num_clients):
            jt = float(self.join_time.get(cid, 0.0))
            if jt <= 0.0:
                # 立即上线
                self._do_join(cid, at_time=0.0, schedule_only=True)
            else:
                self._push_event(jt, "JOIN", cid, self.epoch[cid])

        # 周期评估
        if self.eval_interval > 0:
            self._push_event(self.eval_interval, "EVAL_TICK", None, 0)

    # ----------------------------------------------------------------------
    # 事件循环
    # ----------------------------------------------------------------------
    def run(self, until_time: Optional[float] = None, max_events: Optional[int] = None):
        processed = 0
        while True:
            ev = self._pop_event()
            if ev is None:
                break

            # 终止条件（时间）
            if until_time is not None and ev.due > until_time:
                # 放回以便后续继续
                self._push_event(ev.due, ev.kind, ev.cid, ev.epoch, ev.prio, ev.seq)
                break

            # 推进时间
            self.now = ev.due

            # 懒惰失效：过期事件跳过
            if ev.cid is not None and ev.epoch != self.epoch[ev.cid]:
                continue

            if ev.kind == "JOIN":
                self._handle_join(ev.cid)
            elif ev.kind == "TRAIN_DONE":
                if self._serial:
                    self._handle_train_done_serial(ev.cid)
                else:
                    self._dispatch_train_done_parallel(ev.cid)
            elif ev.kind == "EVAL_TICK":
                if self._serial:
                    self._handle_eval_tick_serial()
                else:
                    self._handle_eval_tick_parallel()
            else:
                raise RuntimeError(f"Unknown event kind: {ev.kind}")

            processed += 1
            if max_events is not None and processed >= max_events:
                break

        # 并行模式下的优雅收尾
        if not self._serial and self._executor is not None:
            self._executor.shutdown(wait=True)

    # ----------------------------------------------------------------------
    # 事件处理器（JOIN）
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
        dur = float(self.all_clients[cid].compute_time_for_next_burst()) * 0.5
        self._push_event(at_time + dur, "TRAIN_DONE", cid, self.epoch[cid])

    # ----------------------------------------------------------------------
    # 串行模式：TRAIN_DONE / EVAL_TICK
    # ----------------------------------------------------------------------
    def _handle_train_done_serial(self, cid: int):
        client = self.all_clients[cid]

        buf = getattr(client, "neighbor_model_weights", None)
        if buf is not None and len(buf) > 0:
            client.aggregate()

        # 1) 本地训练
        client.train()
        # 2) 更新异步元信息
        client.on_train_done(self.now)
        # 3) 选择目标邻居并发送
        receivers = self._sample_online_neighbors(cid, self.k_push)
        payload = client.send_model()
        for r in receivers:
            self.all_clients[r].receive_neighbor_model(payload)
        # 4) 排程下一次 TRAIN_DONE
        dur = float(client.compute_time_for_next_burst()) * 0.5
        self._push_event(self.now + dur, "TRAIN_DONE", cid, self.epoch[cid])

    def _handle_eval_tick_serial(self):
        if self.eval_interval <= 0:
            return

        online_ids = [i for i, on in enumerate(self.online) if on]
        if len(online_ids) > 0:
            client_results: Dict[int, Dict[str, float]] = {}
            agg: Dict[str, float] = {}
            for cid in online_ids:
                metrics = self.all_clients[cid].evaluate_model()
                client_results[cid] = metrics
                for k, v in metrics.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
            overall = {k: (v / len(online_ids)) for k, v in agg.items()}
            # 可记录 overall / client_results
            for cid in online_ids:
                self.all_clients[cid].update_lr()

        self._push_event(self.now + self.eval_interval, "EVAL_TICK", None, 0)

    # ----------------------------------------------------------------------
    # 并行模式：TRAIN_DONE / EVAL_TICK
    # ----------------------------------------------------------------------
    def _dispatch_train_done_parallel(self, cid: int):
        """并行提交训练任务；回调里安排下一次 TRAIN_DONE。"""
        with self._lock:
            if cid in self._inflight:
                return  # 已在跑，忽略重复
            # 开屏障期间不应提交新任务（通常不会收到此事件，但稳妥起见检查）
            if self._eval_barrier:
                return
            fut = self._executor.submit(self._train_task_parallel, cid, self.now)
            self._inflight[cid] = fut
            fut.add_done_callback(lambda f, _cid=cid: self._on_train_future_done(_cid, f))

    def _train_task_parallel(self, cid: int, now_snap: float):
        """
        在线程池中运行的训练任务：
        - 聚合接收缓冲
        - train()
        - on_train_done()
        - 发送给邻居
        - 计算 next_due（若评估屏障已打开，则返回 None）
        """
        client = self.all_clients[cid]

        buf = getattr(client, "neighbor_model_weights", None)
        if buf is not None and len(buf) > 0:
            client.aggregate()

        client.train()
        client.on_train_done(now_snap)

        receivers = self._sample_online_neighbors_threadsafe(cid, self.k_push)
        payload = client.send_model()
        for r in receivers:
            self.all_clients[r].receive_neighbor_model(payload)

        # 屏障期间不排下一次
        if self._eval_barrier:
            return None

        dur = float(client.compute_time_for_next_burst()) * 0.5
        return now_snap + dur

    def _on_train_future_done(self, cid: int, fut: Future):
        try:
            next_due = fut.result()
        except Exception:
            next_due = None

        with self._lock:
            self._inflight.pop(cid, None)
            if self._eval_barrier:
                # 屏障中完成，不做任何排程；评估结束后会统一给每个在线客户端排下一轮
                return
            if next_due is not None:
                self._push_event(next_due, "TRAIN_DONE", cid, self.epoch[cid])

    def _handle_eval_tick_parallel(self):
        """评估屏障：阻断新训练排程，等待在途任务结束，评估后统一排下一轮。"""
        if self.eval_interval <= 0:
            return

        # 1) 打开屏障
        with self._lock:
            self._eval_barrier = True

        # 2) 等待所有在途训练完成
        inflight = []
        with self._lock:
            inflight = list(self._inflight.values())
        for f in inflight:
            try:
                f.result()
            except Exception:
                pass
        with self._lock:
            self._inflight.clear()

        # 3) 评估与 LR 调度
        online_ids = [i for i, on in enumerate(self.online) if on]
        if len(online_ids) > 0:
            client_results: Dict[int, Dict[str, float]] = {}
            agg: Dict[str, float] = {}
            for cid in online_ids:
                metrics = self.all_clients[cid].evaluate_model()
                client_results[cid] = metrics
                for k, v in metrics.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
            overall = {k: (v / len(online_ids)) for k, v in agg.items()}
            # 可记录 overall / client_results
            for cid in online_ids:
                self.all_clients[cid].update_lr()

        # 4) 关闭屏障，并统一为在线客户端排下一轮 TRAIN_DONE
        with self._lock:
            self._eval_barrier = False
            for cid in online_ids:
                dur = float(self.all_clients[cid].compute_time_for_next_burst()) * 0.5
                self._push_event(self.now + dur, "TRAIN_DONE", cid, self.epoch[cid])

        # 5) 排下一次评估
        self._push_event(self.now + self.eval_interval, "EVAL_TICK", None, 0)

    # ----------------------------------------------------------------------
    # 邻居采样
    # ----------------------------------------------------------------------
    def _sample_online_neighbors(self, cid: int, k: int) -> List[int]:
        """串行模式使用：在“出邻居 ∩ 已上线”中均匀无放回采样至多 k 个。"""
        candidates = [j for j in range(self.num_clients) if j != cid and self.online[j]]
        n = len(candidates)
        if n == 0:
            return []
        if k >= n:
            return candidates[:]  # 返回一个副本
        return self._rng.sample(candidates, k)

    def _sample_online_neighbors_threadsafe(self, cid: int, k: int) -> List[int]:
        """并行模式使用（加锁，保证 RNG 的可复现）。"""
        with self._lock:
            candidates = [j for j in range(self.num_clients) if j != cid and self.online[j]]
            n = len(candidates)
            if n == 0:
                return []
            if k >= n:
                return candidates[:]
            return self._rng.sample(candidates, k)

    # ----------------------------------------------------------------------
    # 事件工具（线程安全封装但串行也可用）
    # ----------------------------------------------------------------------
    def _push_event(self, due: float, kind: str, cid: Optional[int], epoch: int,
                    prio: Optional[int] = None, seq: Optional[int] = None):
        if prio is None:
            prio = self._PRIO[kind]
        if seq is None:
            seq = self._next_seq()
        ev = _Event(
            due=float(due),
            prio=int(prio),
            seq=int(seq),
            kind=kind,
            cid=cid,
            epoch=int(epoch),
        )
        with self._heap_lock:
            heapq.heappush(self._heap, ev)

    def _pop_event(self) -> Optional[_Event]:
        with self._heap_lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq_counter += 1
            return self._seq_counter
