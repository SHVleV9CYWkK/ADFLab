from __future__ import annotations

import heapq
import os
import random
import threading
from copy import deepcopy
from dataclasses import dataclass
from queue import Empty
from typing import Dict, List, Optional, Any

import numpy as np
import torch
# 导入Pytorch的多进程库
from torch.multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm  # 导入 tqdm


# --- 辅助函数：必须在顶层，以便子进程可以导入和调用 ---

def _clone_and_detach(tensor_dict: Any) -> Any:
    """
    (已修复)
    递归地克隆、分离张量并将其移动到 CPU。
    这会破坏任何 PyTorch 共享内存句柄，允许张量被安全地 "接力"。
    """
    if isinstance(tensor_dict, dict):
        return {k: _clone_and_detach(v) for k, v in tensor_dict.items()}
    elif isinstance(tensor_dict, list):
        return [_clone_and_detach(v) for v in tensor_dict]
    elif isinstance(tensor_dict, tuple):  # <--- 修复: 增加对 tuple 的支持
        return tuple(_clone_and_detach(v) for v in tensor_dict)
    elif hasattr(tensor_dict, 'clone'):
        try:
            # 即使它已经在 CPU 上，我们仍然 .clone()
            # 来创建一个没有共享历史的新张量。
            return tensor_dict.clone().detach().cpu()
        except Exception:
            # 可能不是一个张量
            return deepcopy(tensor_dict)
    else:
        # 其他可序列化类型 (int, float, str)
        return deepcopy(tensor_dict)


def _worker_process_loop(
        worker_id: int,
        client_id_list: List[int],  # 此 worker 负责的客户端ID列表
        task_queue: Queue,
        result_queue: Queue,
        seed: int,
        # --- 创建 Client 所需的参数 ---
        args: Any,
        all_client_indices: Dict[int, Dict[str, str]],
        init_model_state: Dict[str, torch.Tensor],
        device_str: str,
        time_scale: float,
        join_time: Dict[int, float],  # 接收 join_time 字典
):
    """
    (混合模式) Worker 进程循环。
    它在内部创建和管理 *一组* Client 实例。
    """
    try:
        # --- 1. 初始化环境 ---
        worker_seed = seed + worker_id
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

        device = torch.device(device_str)

        # (已修复) 正确设置 CUDA 设备
        if device.type == "cuda":
            device_index = device.index if device.index is not None else 0
            torch.cuda.set_device(device_index)
            torch.cuda.manual_seed_all(worker_seed)

        # --- 2. Worker 独立加载数据集 ---
        from clients.client_factory import _pick_client_class, _base_hyperparams
        from utils.utils import load_model, load_dataset
        from clients.client import Client  # 导入基类用于类型提示

        # (已修复) Worker 独立加载数据集
        # print(f"[Worker {worker_id}] Loading dataset '{args.dataset_name}'...")
        full_dataset = load_dataset(args.dataset_name)
        # print(f"[Worker {worker_id}] Dataset loaded.")

        # 3. 准备基础超参数 (复刻 client_factory 逻辑)
        client_class = _pick_client_class(args.fl_method)
        train_hyperparam_base = _base_hyperparams(args)

        if getattr(args, "mode", "async_fl") == "async_fl" and args.fl_method == "dfedpgp":
            raise NotImplementedError("dfedpgp is only supported in SYNC mode.")

        # --- 4. 初始化所有此 worker 负责的 clients ---
        managed_clients: Dict[int, Client] = {}
        client_is_online: Dict[int, bool] = {}

        if not client_id_list:
            print(f"[Worker {worker_id}] No clients assigned. Standing by.")

        for cid in client_id_list:
            # 在子进程中为每个 client 创建模型
            worker_model = load_model(args.model, num_classes=len(full_dataset.classes))
            worker_model.load_state_dict(init_model_state)
            worker_model.to(device)

            # (已修复) 复刻 create_client 循环体内的逻辑
            train_hyperparam = deepcopy(train_hyperparam_base)

            # 添加特定于方法的超参数
            if "dfedcad" in args.fl_method or "adfedmac" in args.fl_method:
                train_hyperparam['lambda_kd'] = args.lambda_kd
                train_hyperparam['n_clusters'] = args.n_clusters
                train_hyperparam['lambda_alignment'] = args.lambda_alignment
                train_hyperparam['base_decay_rate'] = args.base_decay_rate
            elif args.fl_method == "dfedmtkdrl":
                train_hyperparam['lambda_kd'] = args.lambda_kd
            elif args.fl_method == "dfedmtkd":
                train_hyperparam['lambda_kd'] = args.lambda_kd
            elif args.fl_method == "dfedsam":
                train_hyperparam['rho'] = args.rho
            elif args.fl_method == "fedgo":
                train_hyperparam['lambda_kd'] = args.lambda_kd

            # 设置延迟标志
            train_hyperparam['is_delayed'] = cid in join_time

            # 实例化 Client
            client: Client = client_class(
                cid,
                all_client_indices[cid],
                full_dataset,
                train_hyperparam,
                device
            )

            client.set_init_model(worker_model)
            managed_clients[cid] = client
            client_is_online[cid] = False

        print(f"[Worker {worker_id}] Successfully initialized {len(managed_clients)} clients.")

        # --- 5. 任务处理循环 ---
        while True:
            task = task_queue.get()
            if task is None or task == "STOP":
                break

            kind, cid, payload = task

            try:
                client = managed_clients[cid]
                is_online = client_is_online[cid]

                if kind == "JOIN":
                    client.init_client()
                    client_is_online[cid] = True
                    dur = float(client.compute_time_for_next_burst()) * time_scale
                    result_queue.put(("JOIN_DONE", cid, dur))

                elif kind == "TRAIN" and is_online:
                    models_to_agg = payload.get('models_to_agg', [])
                    for model_state in models_to_agg:
                        client.receive_neighbor_model(model_state)

                    if models_to_agg:
                        client.aggregate()

                    client.train()
                    client.on_train_done(payload['now_snap'])

                    payload_out = client.send_model()
                    detached_payload = _clone_and_detach(payload_out)

                    dur = float(client.compute_time_for_next_burst()) * time_scale
                    receivers = payload.get('receivers', [])
                    result_queue.put(("TRAIN_DONE", cid, detached_payload, dur, receivers))

                elif kind == "EVAL" and is_online:
                    metrics = client.evaluate_model()
                    result_queue.put(("EVAL_DONE", cid, metrics))

                elif kind == "LR_UPDATE" and is_online:
                    client.update_lr()

            except Exception as e:
                print(f"[Worker {worker_id}, Client {cid}] Error processing {kind}: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put(("WORKER_ERROR", cid, str(e)))

    except Exception as e:
        print(f"[Worker {worker_id}] FATAL: Failed to initialize. {e}")
        import traceback
        traceback.print_exc()
        cid_rep = client_id_list[0] if client_id_list else -1
        result_queue.put(("WORKER_DIED", cid_rep, str(e)))

    finally:
        try:
            task_queue.close()
            result_queue.close()
            print(f"[Worker {worker_id}] Cleanly exiting.")
        except Exception as e:
            # 如果队列已损坏，也无妨
            print(f"[Worker {worker_id}] Error during queue cleanup: {e}")


@dataclass(order=True)
class _Event:
    due: float
    prio: int
    seq: int
    kind: str  # 'JOIN' | 'TRAIN_DONE' | 'EVAL_TICK'
    cid: Optional[int]
    epoch: int


class AsyncCoordinator:
    """
    异步去中心化协调器（混合进程池版）：

    - 启动 `max_workers` 个进程。
    - `num_clients` 个客户端被分配到这些进程中。
    - 主进程只负责事件调度和消息路由。
    """

    _PRIO = {"JOIN": 1, "TRAIN_DONE": 2, "EVAL_TICK": 3}

    def __init__(self,
                 num_clients: int,
                 model_template: torch.nn.Module,
                 all_client_indices: Dict[int, Dict[str, str]],
                 log_queue: Queue,
                 device: torch.device,
                 client_delay_dict: Dict[int, float],
                 args,
                 max_workers: int = 1,
                 total_time: float = 60.0):  # <--- (已添加) 接收 total_time

        try:
            set_start_method('spawn', force=True)
        except RuntimeError as e:
            print(f"Info: Multiprocessing start method already set. {e}")

        # ---- 基本状态 ----
        self.num_clients = num_clients
        self.device = device
        self.log_queue = log_queue
        self.args = args

        self.join_time: Dict[int, float] = {i: 0.0 for i in range(self.num_clients)}
        if client_delay_dict:
            self.join_time.update({int(k): float(v) for k, v in client_delay_dict.items()})

        # 超参
        self.k_push = int(getattr(args, "k_push", 2))
        self.eval_interval = float(getattr(args, "eval_interval", 0.0))
        self.time_scale = float(getattr(args, "train_time_scale", 0.2))

        # 随机性
        self.seed = int(getattr(args, "seed", 0))
        self._rng = random.Random(self.seed)

        # ---- 主进程状态（Worker的镜像） ----
        self.online: List[bool] = [False] * self.num_clients
        self.epoch: List[int] = [0] * self.num_clients

        # 事件堆
        self._heap: List[_Event] = []
        self._seq_counter: int = 0
        self.now: float = 0.0

        # ---- 多进程控制 ----
        if max_workers is None or max_workers < 1:
            max_workers = min(self.num_clients or 1, (os.cpu_count() or 4))

        self.num_workers = min(max_workers, self.num_clients)
        print(f"Initializing AsyncCoordinator with {self.num_workers} workers for {self.num_clients} clients.")

        # ClientID -> WorkerID 的映射
        self.client_to_worker: Dict[int, int] = {
            cid: cid % self.num_workers for cid in range(self.num_clients)
        }
        # WorkerID -> List[ClientID] 的映射
        self.worker_client_map: List[List[int]] = [[] for _ in range(self.num_workers)]
        for cid, wid in self.client_to_worker.items():
            self.worker_client_map[wid].append(cid)

        # 每个 Worker 一个任务队列
        self._task_queues: List[Queue] = [Queue() for _ in range(self.num_workers)]
        self._result_queue: Queue = Queue()
        self._workers: List[Process] = []

        # 主进程中的邻居模型缓冲区
        self._neighbor_buffers: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(self.num_clients)]

        # 在途任务（cid），用于评估屏障
        self._inflight: set[int] = set()
        self._eval_barrier: bool = False

        # 评估屏障状态
        self._eval_pending_clients: set[int] = set()
        self._eval_client_results: Dict[int, Dict[str, float]] = {}

        # 性能跟踪 (用于日志)
        self._today_date = getattr(args, "today_date", "1970-01-01")
        self._exper_num = getattr(args, "exper_num", 0)
        self._eval_tick_counter = 0

        # --- 启动 Workers ---
        print(f"Spawning {self.num_workers} worker processes...")
        init_model_state = _clone_and_detach(model_template.state_dict())
        device_str = str(device)

        for wid in range(self.num_workers):
            client_id_list = self.worker_client_map[wid]
            p = Process(
                target=_worker_process_loop,
                args=(
                    wid,
                    client_id_list,
                    self._task_queues[wid],
                    self._result_queue,
                    self.seed,
                    self.args,
                    all_client_indices,
                    # (已修复) 移除 full_dataset
                    init_model_state,
                    device_str,
                    self.time_scale,
                    self.join_time,  # (已修复) 传入 join_time
                ),
                daemon=True
            )
            p.start()
            self._workers.append(p)

        # --- (已添加) 初始化进度条 ---
        self.pbar = tqdm(total=total_time, desc="Async Sim", unit=" t")
        self.last_pbar_update_time: float = 0.0
        # --- (结束) ---

        self._bootstrap_events()

    # ----------------------------------------------------------------------
    # 事件循环 (已修复终止Bug 和 添加进度条)
    # ----------------------------------------------------------------------
    def run(self, until_time: Optional[float] = None):
        """运行事件循环直到 specified time。"""

        target_time = float('inf') if until_time is None else until_time

        try:
            while self.now < target_time:

                # --- (已添加) 更新进度条 ---
                dt = self.now - self.last_pbar_update_time
                if dt > 0.001:
                    self.pbar.update(dt)
                    self.last_pbar_update_time = self.now
                # --- (结束) ---

                # 1. 非阻塞地处理所有已到达的结果
                while True:
                    try:
                        result = self._result_queue.get_nowait()
                        self._handle_worker_result(result)
                    except Empty:
                        break

                        # 2. 查看下一个事件
                ev = self._peek_event()

                # 3. 确定下一步行动
                if ev is None:
                    # --- (已修复) 终止条件 ---
                    if not self._inflight and not self._eval_pending_clients:
                        print(
                            "\n[Coordinator] Event heap empty, no inflight tasks, and no pending eval. Simulation ended.")
                        self.now = target_time
                        break
                    # --- (修复结束) ---

                    # 堆空了，但我们仍在忙碌 (训练或评估)，所以必须阻塞等待
                    wait_time = max(0, target_time - self.now)
                    if wait_time == 0: break

                    try:
                        result = self._result_queue.get(timeout=wait_time)
                        self._handle_worker_result(result)
                        continue
                    except Empty:
                        # 超时，时间到达 target_time
                        self.now = target_time
                        break

                # 4. 如果下一个事件在未来
                if ev.due > self.now:
                    wait_time = min(ev.due, target_time) - self.now

                    try:
                        result = self._result_queue.get(timeout=wait_time)
                        self._handle_worker_result(result)
                        continue
                    except Empty:
                        self.now = min(ev.due, target_time)

                # 5. 如果时间已到 target_time，停止
                if self.now >= target_time:
                    break

                # 6. 时间到达 ev.due，处理事件
                ev = self._pop_event()

                if ev.cid is not None and ev.epoch != self.epoch[ev.cid]:
                    continue

                if ev.kind == "JOIN":
                    self._handle_join(ev.cid)
                elif ev.kind == "TRAIN_DONE":
                    self._handle_train_done(ev.cid)
                elif ev.kind == "EVAL_TICK":
                    self._handle_eval_tick()
        finally:
            # --- (已添加) 最终更新进度条 ---
            dt_final = target_time - self.last_pbar_update_time
            if dt_final > 0:
                self.pbar.update(dt_final)
            # --- (结束) ---

    # ----------------------------------------------------------------------
    # 事件处理器
    # ----------------------------------------------------------------------

    def _bootstrap_events(self):
        for cid in range(self.num_clients):
            jt = float(self.join_time.get(cid, 0.0))
            self._push_event(jt, "JOIN", cid, self.epoch[cid])

        if self.eval_interval > 0:
            self._push_event(self.eval_interval, "EVAL_TICK", None, 0)
        else:
            print("[Coordinator] eval_interval <= 0, 内部评估已禁用。")

    def _handle_join(self, cid: int):
        """主进程：向 *负责的 Worker* 发送 JOIN 任务。"""
        self.epoch[cid] += 1
        self.online[cid] = True  # 乐观假设
        self._inflight.add(cid)

        wid = self.client_to_worker[cid]
        self._task_queues[wid].put(("JOIN", cid, None))

    def _handle_train_done(self, cid: int):
        """(已修复) 主进程：向 *负责的 Worker* 发送 TRAIN 任务。"""
        if cid in self._inflight:
            return
        if self._eval_barrier:
            return

        models_to_agg_from_buffer = self._neighbor_buffers[cid]
        self._neighbor_buffers[cid] = []
        receivers = self._sample_online_neighbors(cid, self.k_push)

        # --- (已修复) 必须克隆这些“中继”张量 ---
        models_to_send = _clone_and_detach(models_to_agg_from_buffer)

        payload = {
            'models_to_agg': models_to_send,  # <--- 发送克隆后的副本
            'receivers': receivers,
            'now_snap': self.now
        }

        self._inflight.add(cid)

        wid = self.client_to_worker[cid]
        self._task_queues[wid].put(("TRAIN", cid, payload))

    def _handle_eval_tick(self):
        """主进程：启动评估屏障。"""
        if self.eval_interval <= 0: return

        self._eval_barrier = True
        online_ids = {i for i, on in enumerate(self.online) if on}
        self._eval_pending_clients = online_ids
        self._eval_client_results = {}

        if not self._inflight and self._eval_pending_clients:
            # print(f"\n[t={self.now:.2f}] EVAL_TICK: 触发 {len(online_ids)} 个客户端评估。")
            for cid in self._eval_pending_clients:
                wid = self.client_to_worker[cid]
                self._task_queues[wid].put(("EVAL", cid, None))

    # ----------------------------------------------------------------------
    # 结果处理器
    # ----------------------------------------------------------------------

    def _handle_worker_result(self, result: tuple):
        """主进程：处理来自 Worker 的异步结果。"""
        kind, cid, *payload = result

        if kind == "WORKER_ERROR":
            print(f"\n[Main] Worker 报告 Client {cid} 错误: {payload[0]}")
            self._inflight.discard(cid)
            return
        if kind == "WORKER_DIED":
            print(f"\n[Main] Worker 致命错误 (报告为 {cid}): {payload[0]}")
            self.online[cid] = False
            self._inflight.discard(cid)
            return

        if self.epoch[cid] == 0 and kind != "JOIN_DONE":
            return  # 尚未 JOIN，忽略

        if kind == "JOIN_DONE":
            self._inflight.discard(cid)
            dur = payload[0]
            self._push_event(self.now + dur, "TRAIN_DONE", cid, self.epoch[cid])

        elif kind == "TRAIN_DONE":
            self._inflight.discard(cid)
            detached_payload, dur, receivers = payload

            for r_cid in receivers:
                if self.online[r_cid]:
                    self._neighbor_buffers[r_cid].append(detached_payload)

            if not self._eval_barrier:
                self._push_event(self.now + dur, "TRAIN_DONE", cid, self.epoch[cid])

        elif kind == "EVAL_DONE":
            metrics = payload[0]
            if cid in self._eval_pending_clients:
                self._eval_pending_clients.remove(cid)
                self._eval_client_results[cid] = metrics

                if not self._eval_pending_clients:
                    self._finalize_eval()

        # --- 检查评估屏障 ---
        if self._eval_barrier and not self._inflight and self._eval_pending_clients:
            if not self._eval_client_results:
                # print(f"\n[t={self.now:.2f}] EVAL_TICK: 在途训练完成。触发 {len(self._eval_pending_clients)} 个客户端评估。")
                for cid in self._eval_pending_clients:
                    wid = self.client_to_worker[cid]
                    self._task_queues[wid].put(("EVAL", cid, None))

    def _finalize_eval(self):
        client_results = self._eval_client_results
        overall_results = {}
        num_online = len(client_results)

        if client_results:
            agg: Dict[str, float] = {}
            for res in client_results.values():
                for k, v in res.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
            overall_results = {k: (v / len(client_results)) for k, v in agg.items()}

        # 1. 更新进度条后缀
        if overall_results:
            acc = overall_results.get("accuracy_micro", overall_results.get("accuracy", 0.0))
            loss = overall_results.get("loss_micro", overall_results.get("loss", 0.0))
            self.pbar.set_postfix(acc=f"{acc:.4f}", loss=f"{loss:.4f}", online=num_online)
        else:
            self.pbar.set_postfix(acc="N/A", online=num_online)

        # 2. 推送到日志队列
        self.log_queue.put((
            overall_results,
            client_results,
            self._today_date,
            self._exper_num,
            self._eval_tick_counter,
            self.args
        ))
        self._eval_tick_counter += 1

        # 3. 调度 LR 更新
        online_cids = list(client_results.keys())
        for cid in online_cids:
            wid = self.client_to_worker[cid]
            self._task_queues[wid].put(("LR_UPDATE", cid, None))

        # 4. 解除屏障并重新调度训练
        self._eval_barrier = False
        for cid in online_cids:
            jitter_dur = 0.01 + random.random() * 0.1
            self._push_event(self.now + jitter_dur, "TRAIN_DONE", cid, self.epoch[cid])

        # 5. 调度下一次评估 (!!! MODIFIED: 绝对间隔规则 !!!)
        if self.eval_interval > 0:
            next_tick_num = np.floor(self.now / self.eval_interval) + 1

            next_eval_time = next_tick_num * self.eval_interval

            self._push_event(next_eval_time, "EVAL_TICK", None, 0)

        # 6. 清理
        self._eval_client_results = {}
        self._eval_pending_clients = set()

    # ----------------------------------------------------------------------
    # 邻居采样
    # ----------------------------------------------------------------------
    def _sample_online_neighbors(self, cid: int, k: int) -> List[int]:
        """在“出邻居 ∩ 已上线”中均匀无放回采样至多 k 个。"""
        candidates = [j for j in range(self.num_clients) if j != cid and self.online[j]]
        n = len(candidates)
        if n == 0:
            return []
        if k >= n:
            return candidates[:]
        return self._rng.sample(candidates, k)

    # ----------------------------------------------------------------------
    # 事件堆工具
    # ----------------------------------------------------------------------
    def _push_event(self, due: float, kind: str, cid: Optional[int], epoch: int):
        prio = self._PRIO[kind]
        seq = self._next_seq()
        ev = _Event(float(due), int(prio), int(seq), kind, cid, int(epoch))
        heapq.heappush(self._heap, ev)

    def _pop_event(self) -> Optional[_Event]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def _peek_event(self) -> Optional[_Event]:
        if not self._heap:
            return None
        return self._heap[0]

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    # ----------------------------------------------------------------------
    # 终止 (已添加进度条)
    # ----------------------------------------------------------------------
    def shutdown(self):
        """终止所有 Worker 进程。"""

        # --- (已添加) 关闭进度条 ---
        if hasattr(self, 'pbar') and self.pbar:
            # self.pbar.refresh() # 确保最后的状态被打印
            self.pbar.close()
            print("\nSimulation progress bar closed.")  # 添加换行
        # --- (结束) ---

        print(f"Shutting down {self.num_workers} workers...")
        for q in self._task_queues:
            try:
                q.put("STOP")
            except Exception:
                pass  # 队列可能已损坏

        for p in self._workers:
            try:
                p.join(timeout=5.0)
                if p.is_alive():
                    print(f"Worker {p.pid} did not exit gracefully, terminating.")
                    p.terminate()
            except Exception as e:
                print(f"Error shutting down worker: {e}")

        try:
            self._result_queue.close()
        except Exception:
            pass
        print("Shutdown complete.")

    def __del__(self):
        # __del__ 中不应处理多进程资源，非常不可靠
        pass