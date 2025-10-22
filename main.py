import json
import os
import random
import time
from datetime import datetime
from multiprocessing import Queue, Process
from typing import Dict

import numpy as np
import torch
import logging

from tqdm import tqdm

logging.getLogger().setLevel(logging.ERROR)

from clients.client_factory import create_client
from coordinator import Coordinator                  # 同步基线
from async_coordinator import AsyncCoordinator       # 异步新协调器
from evaluator import Evaluator                      # 统一评估
from utils.args import parse_args
from utils.experiment_logger import ExperimentLogger
from utils.utils import (
    load_model, load_dataset, get_client_data_indices,
    get_client_delay_info, save_log, get_experiment_num
)


# --------------------------
# 日志工作进程（与旧版兼容）
# --------------------------
def log_worker(queue: Queue):
    while True:
        item = queue.get()
        if item == "STOP":
            break
        overall_results, client_results, date, exper_num, round_num, args = item
        # overall
        save_log(overall_results, date, exper_num, round_num, args)
        # per-client
        for client_id, client_result in client_results.items():
            save_log(client_result, date, exper_num, round_num, args, client_id)


# --------------------------
# Join 时间表构建
# --------------------------
def build_join_table(num_clients: int, args, logger) -> Dict[int, float]:
    """
    返回 {client_id: join_time_float}。
    优先级：
      1) --join_table 指定 JSON：{ "<id>": <time>, ... }；
      2) 回退到 get_client_delay_info（把“延迟轮次”直接当作时间）。
    """
    if getattr(args, "join_table", None):
        path = args.join_table
        if not os.path.isfile(path):
            raise FileNotFoundError(f"join_table not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        table = {int(k): float(v) for k, v in raw.items()}
        logger.save("join_time", table)
        return table

    # 回退：使用旧的延迟生成逻辑（单位视作时间）
    delay_rounds = get_client_delay_info(
        num_clients,
        getattr(args, "delay_client_ratio", 0.0),
        getattr(args, "minimum_join_rounds", 0),
        getattr(args, "n_rounds", 50),
        getattr(args, "temp_client_dist", "single"),
        getattr(args, "set_single_delay_client", -1),
    )
    table = {int(cid): float(r) for cid, r in delay_rounds.items()}
    logger.save("join_time", table)
    return table


# --------------------------
# 同步执行（轮次驱动）
# --------------------------
def run_sync(coordinator: Coordinator, args, today_date, exper_num):
    log_queue = Queue()
    log_process = Process(target=log_worker, args=(log_queue,))
    log_process.start()

    evaluator = Evaluator(include_micro=True, include_quantiles=True)

    # 交换方法保持兼容
    interchange = coordinator.interchange_model
    if args.fl_method == "dfedpgp":
        interchange = coordinator.interchange_model_dfedpgp

    for r in range(args.n_rounds):
        print(f"Round {r}")
        t0 = time.time()

        coordinator.train_client(r)
        interchange(r)

        # 统一口径评估：同步下，当前参与训练的客户端即“在线集合”
        overall_results, client_results = evaluator.evaluate_online(
            coordinator.participated_training_clients,
            [True] * len(coordinator.participated_training_clients)
        )

        t1 = time.time()
        log_queue.put((overall_results, client_results, today_date, exper_num, r, args))

        if overall_results:
            s = ', '.join([f"{k.capitalize()}: {v:.4f}" for k, v in overall_results.items()])
            print(f"Training time: {(t1 - t0):.2f}. Eval: {s}")
            accs = [res["accuracy"] for res in client_results.values()]
            max_cid = max(client_results, key=lambda cid: client_results[cid]["accuracy"])
            min_cid = min(client_results, key=lambda cid: client_results[cid]["accuracy"])
            print(f"Client Accuracy — Max: {max(accs):.4f} (Client {max_cid}), "
                  f"Min: {min(accs):.4f} (Client {min_cid})")
        else:
            print("No clients evaluated in this round.")

        coordinator.lr_scheduler()
        print("------------")

    log_queue.put("STOP")
    log_process.join()


# --------------------------
# 异步执行（事件驱动 + 时间对齐评估）
# --------------------------
def run_async(coordinator: AsyncCoordinator, args, today_date, exper_num):
    evaluator = Evaluator(include_micro=True, include_quantiles=True)

    eval_interval = float(getattr(args, "eval_interval", 0.0))
    # 优先 total_time，其次 n_rounds
    total_time = float(args.total_time) if getattr(args, "total_time", None) is not None \
                 else float(getattr(args, "n_rounds", 50))

    if total_time <= 0:
        raise ValueError("total_time must be > 0 (or set n_rounds > 0).")
    if eval_interval < 0:
        eval_interval = 0.0
    if eval_interval > 0 and eval_interval > total_time:
        print(f"[Warn] eval_interval ({eval_interval}) > total_time ({total_time}); "
              f"will evaluate only once at the end.")
        eval_interval = 0.0

    log_queue = Queue()
    log_process = Process(target=log_worker, args=(log_queue,))
    log_process.start()

    # 进度条：以“墙钟时间”为单位
    pbar = tqdm(total=total_time, desc="Async time", unit="t")

    next_eval_time = eval_interval if eval_interval > 0 else total_time
    tick = 0
    last_eval_now = coordinator.now

    # 性能跟踪（时间加权平均）
    last_overall = {}
    best_acc = -1.0
    best_time = 0.0
    acc_dt = []   # [(acc, dt), ...] 用于时间加权平均
    online_hist = []  # 在线客户端数的平均参考

    while coordinator.now < total_time and coordinator._heap:
        target = min(next_eval_time, total_time)
        coordinator.run(until_time=target)

        overall_results, client_results = evaluator.evaluate_online(
            coordinator.all_clients, coordinator.online
        )
        log_queue.put((overall_results, client_results, today_date, exper_num, tick, args))

        # 进度条更新（按墙钟时间前进量）
        dt = coordinator.now - last_eval_now
        if dt > 0:
            pbar.update(dt)

        # 展示关键指标
        cur_acc = overall_results.get("accuracy_micro", overall_results.get("accuracy", 0.0))
        cur_loss = overall_results.get("loss_micro", overall_results.get("loss", 0.0))
        num_online = sum(1 for x in coordinator.online if x)
        pbar.set_postfix(t=f"{coordinator.now:.2f}", acc=f"{cur_acc:.4f}",
                         loss=f"{float(cur_loss):.4f}", online=num_online)


        # 跟踪最优与时间加权
        if overall_results and dt > 0:
            acc_dt.append((float(cur_acc), dt))
            online_hist.append(num_online)
            if float(cur_acc) > best_acc:
                best_acc = float(cur_acc)
                best_time = coordinator.now
            last_overall = overall_results

        tick += 1
        last_eval_now = coordinator.now
        if eval_interval > 0:
            next_eval_time += eval_interval
        else:
            break  # 只评一次

    # 若还有剩余时间，推进到总时长并做最终评估
    if eval_interval > 0 and coordinator.now < total_time:
        coordinator.run(until_time=total_time)
        overall_results, client_results = evaluator.evaluate_online(
            coordinator.all_clients, coordinator.online
        )
        log_queue.put((overall_results, client_results, today_date, exper_num, tick, args))
        dt_final = coordinator.now - last_eval_now
        if dt_final > 0 and overall_results:
            acc_dt.append((float(overall_results.get("accuracy_micro",
                                                     overall_results.get("accuracy", 0.0))), dt_final))
            online_hist.append(sum(1 for x in coordinator.online if x))
            last_overall = overall_results
            pbar.update(dt_final)
            pbar.set_postfix(t=f"{coordinator.now:.2f}",
                             acc=f"{overall_results.get('accuracy_micro', overall_results.get('accuracy', 0.0)):.4f}",
                             loss=f"{float(overall_results.get('loss_micro', overall_results.get('loss', 0.0))):.4f}",
                             online=sum(1 for x in coordinator.online if x))

    pbar.close()

    # 总结：时间加权平均 / 最优 / 期末
    if acc_dt:
        total_dt = sum(dt for _, dt in acc_dt)
        twa = sum(acc * dt for acc, dt in acc_dt) / total_dt  # time-weighted average
        final_acc = last_overall.get("accuracy_micro", last_overall.get("accuracy", 0.0))
        mean_online = sum(online_hist) / len(online_hist) if online_hist else 0.0
        print(f"[ASYNC Summary] T={total_time} "
              f"| final_acc={final_acc:.4f} "
              f"| best_acc={best_acc:.4f} @t={best_time:.2f} "
              f"| time_weighted_acc={twa:.4f} "
              f"| mean_online={mean_online:.2f}")

    log_queue.put("STOP")
    log_process.join()

# --------------------------
# 装配与入口
# --------------------------
def main():
    args = parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 设备
    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    today = datetime.today().strftime('%Y-%m-%d')
    exper_num = get_experiment_num(today, args)

    with ExperimentLogger(today, exper_num, device, args) as logger:
        print("Initializing")
        # 数据与模型
        full_dataset = load_dataset(args.dataset_name)
        model = load_model(args.model, num_classes=len(full_dataset.classes)).to(device)
        client_indices, num_clients = get_client_data_indices(
            args.dataset_indexes_dir, args.dataset_name, args.split_method, args.alpha
        )

        # 异步：Join 时间表（单位=时间），并关闭协调器内部评估
        join_time = build_join_table(num_clients, args, logger)

        # 客户端
        clients = create_client(num_clients, args, client_indices, full_dataset, join_time.keys(), device)

        if args.mode == "sync_fl":
            # 同步：沿用旧的延迟机制（按轮次）
            delay_rounds = get_client_delay_info(
                num_clients, args.delay_client_ratio, args.minimum_join_rounds, args.n_rounds,
                args.temp_client_dist, args.set_single_delay_client
            )
            logger.save("client_delay", delay_rounds)
            coordinator = Coordinator(clients, model, device, delay_rounds, args)
            run_sync(coordinator, args, today, exper_num)
        else:

            # 创建一个 args 副本，把 eval_interval 置 0（防止协调器内部也评估）
            class _NS: pass
            args_async = _NS()
            for k, v in vars(args).items():
                setattr(args_async, k, v)
            args_async.eval_interval = 0.0
            if args_async.k_push is None:
                args_async.k_push = args_async.num_conn

            coordinator = AsyncCoordinator(clients, model, device, join_time, args_async)
            print("Completed initialization")
            print("Start training")
            run_async(coordinator, args, today, exper_num)

        print("Done")


if __name__ == '__main__':
    main()