import json
import os
import random
import time
from datetime import datetime
# 导入 torch.multiprocessing
import torch.multiprocessing as mp
from typing import Dict

import numpy as np
import torch
import logging

from tqdm import tqdm

logging.getLogger().setLevel(logging.ERROR)

from clients.client_factory import create_client
from coordinator import Coordinator  # 同步基线
from async_coordinator import AsyncCoordinator  # 异步新协调器
from evaluator import Evaluator  # 统一评估
from utils.args import parse_args
from utils.experiment_logger import ExperimentLogger
from utils.utils import (
    load_model, load_dataset, get_client_data_indices,
    get_client_delay_info, save_log, get_experiment_num
)


# --------------------------
# 日志工作进程（不变）
# --------------------------
def log_worker(queue: mp.Queue):
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
# Join 时间表构建（不变）
# --------------------------
def build_join_table(num_clients: int, args, logger) -> Dict[int, float]:
    """
    返回 {client_id: join_time_float}。
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
# 同步执行（不变）
# --------------------------
def run_sync(coordinator: Coordinator, args, today_date, exper_num):
    log_queue = mp.Queue()
    log_process = mp.Process(target=log_worker, args=(log_queue,))
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

        # 统一口径评估
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
# 异步执行 (!!! 已移除 !!!)
# --------------------------
# `run_async` 函数已被移除。
# 协调器现在通过 eval_interval 自动处理评估和日志记录。


# --------------------------
# 装配与入口
# --------------------------
def main():
    args = parse_args()

    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Info: Multiprocessing start method already set or error: {e}")

    # ... (种子, 设备, 日期, ... 保持不变) ...
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
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
        # --- (!!! MODIFIED: 移除 full_dataset !!!) ---
        # 我们让 worker 自己加载，以避免 CUDA 错误
        # full_dataset = load_dataset(args.dataset_name)

        # (获取数据集类数，用于模型加载)
        if args.dataset_name == 'cifar10':
            num_classes = 10
        elif args.dataset_name == 'cifar100':
            num_classes = 100
        elif args.dataset_name == 'emnist':
            num_classes = 62
        elif args.dataset_name == 'tiny_imagenet':
            num_classes = 200
        elif args.dataset_name == 'mnist':
            num_classes = 10
        else:
            raise ValueError(f"Unknown dataset {args.dataset_name}")

        model = load_model(args.model, num_classes=num_classes).to(device)
        client_indices, num_clients = get_client_data_indices(
            args.dataset_indexes_dir, args.dataset_name, args.split_method, args.alpha
        )

        if args.mode == "sync_fl":
            # --- 同步模式 (不变, 但需要加载 full_dataset) ---
            print("Loading dataset for sync mode...")
            full_dataset = load_dataset(args.dataset_name)  # 同步模式在主进程加载
            delay_rounds = get_client_delay_info(
                num_clients, args.delay_client_ratio, args.minimum_join_rounds, args.n_rounds,
                args.temp_client_dist, args.set_single_delay_client
            )
            logger.save("client_delay", delay_rounds)
            clients = create_client(num_clients, args, client_indices, full_dataset, list(client_indices.keys()),
                                    device)
            coordinator = Coordinator(clients, model, device, delay_rounds, args)
            run_sync(coordinator, args, today, exper_num)

        else:
            # --- 异步模式 (修改) ---
            join_time = build_join_table(num_clients, args, logger)

            log_queue = mp.Queue()
            log_process = mp.Process(target=log_worker, args=(log_queue,))
            log_process.start()

            args.today_date = today
            args.exper_num = exper_num
            if args.k_push is None:
                args.k_push = args.num_conn

            # --- (!!! MODIFIED: 1. 在此处计算 total_time !!!) ---
            total_time = float(args.total_time) if getattr(args, "total_time", None) is not None \
                else float(getattr(args, "n_rounds", 50))
            if total_time <= 0:
                raise ValueError("total_time (or n_rounds fallback) must be > 0.")
            # --- (!!! 结束 !!!) ---

            # 4. 初始化多进程协调器
            coordinator = AsyncCoordinator(
                num_clients=num_clients,
                model_template=model,
                all_client_indices=client_indices,
                log_queue=log_queue,
                device=device,
                client_delay_dict=join_time,
                args=args,
                max_workers=args.n_job,
                total_time=total_time  # <--- 2. 传入 total_time
            )

            print("Completed initialization")
            print("Start training")

            # 5. (!!! MODIFIED: 移除这里的 print !!!) ---
            # print(f"Running async simulation for {total_time} virtual time units...")

            # 6. 单次调用 run (不变)
            coordinator.run(until_time=total_time)

            # 7. 关闭协调器 (不变)
            coordinator.shutdown()

            # 8. 关闭日志进程 (不变)
            log_queue.put("STOP")
            log_process.join()

        print("Done")
        return


if __name__ == '__main__':
    main()