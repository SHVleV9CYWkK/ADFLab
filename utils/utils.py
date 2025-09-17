# utils/utils.py
from __future__ import annotations

import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, EMNIST, MNIST
from transformers import MobileBertForSequenceClassification

from models.cnn_model import CNNModel, LeafCNN1, LeNet, AlexNet, ResNet18, VGG16, ResNet50

__all__ = [
    "load_dataset",
    "load_model",
    "get_client_data_indices",
    "get_optimizer",
    "get_lr_scheduler",
    "get_client_delay_info",
    "save_log",
    "get_experiment_num",
    # optional helpers
    "seed_all",
    "model_num_bytes",
]


# -----------------------------
# Optional helpers
# -----------------------------
def seed_all(seed: int) -> None:
    """Set random seeds for reproducibility (Python, NumPy, PyTorch CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def model_num_bytes(model: torch.nn.Module) -> int:
    """Estimate parameter size in bytes (state_dict only; no optimizer buffers)."""
    total = 0
    for p in model.state_dict().values():
        total += p.numel() * p.element_size()
    return int(total)


# -----------------------------
# Dataset / Model loaders
# -----------------------------
def load_dataset(dataset_name: str):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)

    elif dataset_name == 'emnist':
        # Keep compatibility with your previous URL override
        url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
        if url != EMNIST.url:
            print('The URL of the dataset is inconsistent with the latest URL')
            EMNIST.url = url
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = EMNIST(root='./data', train=True, download=True, transform=transform, split="byclass")

    elif dataset_name == 'tiny_imagenet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)

    elif dataset_name == 'mnist':
        transform = transforms.ToTensor()
        dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    else:
        raise ValueError(f"dataset_name does not contain {dataset_name}")
    return dataset


def load_model(model_name: str, num_classes: int):
    if model_name == 'alexnet':
        model = AlexNet(num_classes)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes)
    elif model_name == 'vgg16':
        model = VGG16(num_classes)
    elif model_name == 'cnn':
        model = CNNModel(num_classes)
    elif model_name == 'leafcnn1':
        model = LeafCNN1(num_classes)
    elif model_name == 'lenet':
        model = LeNet(num_classes)
    elif model_name == 'mobilebart':
        model = MobileBertForSequenceClassification.from_pretrained(
            "lordtt13/emo-mobilebert",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # Return logits only for a classifier-like interface
        original_forward = model.forward

        def forward_with_logits_only(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            return outputs.logits

        model.forward = forward_with_logits_only
    else:
        raise ValueError(f"model_name does not contain {model_name}")
    return model


# -----------------------------
# Dataset split indices
# -----------------------------
def get_client_data_indices(root_dir: str, dataset_name: str,
                            split_method: str, alpha: float) -> Tuple[Dict[int, Dict[str, str]], int]:
    """
    Return (client_indices_map, num_clients)
    client_indices_map[cid] = {'train': path_to_npy, 'val': path_to_npy}
    """
    dir_path = os.path.join(root_dir, f"{dataset_name}_{split_method}_{alpha}")

    if not os.path.exists(dir_path):
        dir_path = os.path.join(root_dir, f"{dataset_name}_{split_method}")
        if not os.path.exists(dir_path):
            raise ValueError(f"No matching dataset and split method found for {dataset_name} and {split_method}")

    # client subfolders (e.g., client_0, client_1, ...)
    client_dirs = sorted([
        d for d in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, d))
    ])

    num_clients = len(client_dirs)
    if num_clients == 0:
        raise ValueError(f"No client folders found under {dir_path}")

    client_indices: Dict[int, Dict[str, str]] = {}
    for client_dir in client_dirs:
        # expect folder name like 'client_0'
        try:
            client_id = int(client_dir.split('_')[-1])
        except Exception:
            raise ValueError(f"Unexpected client folder name: {client_dir}")
        client_indices[client_id] = {
            'train': os.path.join(dir_path, client_dir, 'train_indexes.npy'),
            'val': os.path.join(dir_path, client_dir, 'val_indexes.npy'),
        }

    return client_indices, num_clients


# -----------------------------
# Optimizer / Scheduler
# -----------------------------
def get_optimizer(optimizer_name: str, parameters, lr: float):
    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr)
    else:
        raise NotImplementedError(f"{optimizer_name} optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name: str, n_rounds: int | None = None, gated_learner: bool = False):
    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        if gated_learner:
            milestones = [3 * (n_rounds // 4)]
        else:
            milestones = [n_rounds // 2, 3 * (n_rounds // 4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    elif "reduce_on_plateau" in scheduler_name:
        last_word = scheduler_name.split("_")[-1]
        patience = int(last_word) if last_word.isdigit() else 10
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=0.75)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")


# -----------------------------
# Delayed join info (legacy; used as join times in async_fl)
# -----------------------------
def get_client_delay_info(num_clients: int, delay_client_ratio: float,
                          minimum_round: int, total_rounds: int,
                          dist_type: str = "single", preset_client_id: int = -1,
                          *, time_mode: bool = False) -> dict:
    """
    Generate delayed joins for a subset of clients.

    When time_mode == False (legacy, default):
        - Interpret minimum_round and total_rounds as "round indices".
        - Return {client_id: join_round (int)}.

    When time_mode == True (async wall-clock mode):
        - Interpret minimum_round -> min_time (float), total_rounds -> max_time (float).
        - Return {client_id: join_time (float)} where join_time ∈ [min_time, max_time].
          Non-delayed clients are not included (caller should default their join_time to 0.0).

    Args:
        num_clients: total number of clients
        delay_client_ratio: fraction of clients that are delayed (>0 joins later)
        minimum_round: (legacy) earliest round allowed for delay
                       (time_mode=True) earliest join time (min_time)
        total_rounds: (legacy) latest round index (inclusive)
                      (time_mode=True) latest join time (max_time)
        dist_type: 'single' | 'uniform' | 'even' | 'normal'
        preset_client_id: (for 'single') specify which client is delayed; -1 means random
        time_mode: if True, generate continuous times; else generate integer rounds
    Returns:
        dict mapping delayed client_id -> join_round (int) or join_time (float)
    """
    if not (0 <= delay_client_ratio <= 1):
        raise ValueError("delay_client_ratio must be between 0 and 1")

    # ---- Parameter naming for clarity ----
    if time_mode:
        min_time = float(minimum_round)
        max_time = float(total_rounds)
        if max_time <= min_time:
            raise ValueError("In time_mode=True, max_time must be > min_time")
    else:
        if total_rounds <= minimum_round:
            raise ValueError("total_rounds must be greater than minimum_round")

    client_ids = list(range(num_clients))
    delayed: dict = {}

    # ---- 'single' -> delay exactly one client ----
    if dist_type.lower() == "single":
        if num_clients < 1:
            raise ValueError("There must be at least one client to delay.")
        target_cid = preset_client_id if preset_client_id > -1 else random.choice(client_ids)
        if time_mode:
            delayed[target_cid] = float(min_time)
        else:
            delayed[target_cid] = int(minimum_round)
        print(f"Delayed clients:{delayed}")
        return delayed

    # ---- choose delayed set ----
    num_delayed = int(round(num_clients * delay_client_ratio))
    if num_delayed <= 0:
        print("Delayed clients:{}")
        return {}

    delayed_client_ids = set(random.sample(client_ids, num_delayed))

    # ---- distribution over the support ----
    if dist_type.lower() == "uniform":
        for cid in delayed_client_ids:
            if time_mode:
                join_time = random.uniform(min_time, max_time)
                delayed[cid] = float(join_time)
            else:
                join_round = random.randint(minimum_round + 1, total_rounds)
                delayed[cid] = int(join_round)

    elif dist_type.lower() == "even":
        if time_mode:
            # evenly spaced times across [min_time, max_time]
            if num_delayed == 1:
                times = [min_time]
            else:
                # np.linspace returns numpy floats; keep as python float
                times = [float(x) for x in np.linspace(min_time, max_time, num=num_delayed, endpoint=True)]
            random.shuffle(times)
            for cid in delayed_client_ids:
                delayed[cid] = times.pop()
        else:
            available_rounds = list(range(minimum_round + 1, total_rounds + 1))
            nR = len(available_rounds)
            base = num_delayed // nR
            rem = num_delayed % nR
            delayed_rounds_list = []
            for r in available_rounds:
                delayed_rounds_list.extend([r] * base)
            extra_rounds = random.sample(available_rounds, rem) if rem > 0 else []
            delayed_rounds_list.extend(extra_rounds)
            random.shuffle(delayed_rounds_list)
            for cid in delayed_client_ids:
                delayed[cid] = delayed_rounds_list.pop() if delayed_rounds_list else minimum_round

    elif dist_type.lower() == "normal":
        if time_mode:
            mu = 0.5 * (min_time + max_time)
            sigma = (max_time - min_time) / 4.0 if max_time > min_time else 1.0
            for cid in delayed_client_ids:
                # truncated normal via rejection sampling
                while True:
                    x = random.gauss(mu, sigma)
                    if min_time <= x <= max_time:
                        delayed[cid] = float(x)
                        break
        else:
            mu = (minimum_round + 1 + total_rounds) / 2.0
            sigma = (total_rounds - minimum_round) / 4.0
            for cid in delayed_client_ids:
                while True:
                    r = int(round(random.gauss(mu, sigma)))
                    if minimum_round + 1 <= r <= total_rounds:
                        delayed[cid] = r
                        break
    else:
        raise ValueError("Unidentified distribution type, use 'uniform', 'even', or 'normal' (or 'single').")

    print(f"Delayed clients:{delayed}")
    return delayed

# -----------------------------
# Logging helpers (compatible with your existing pipeline)
# -----------------------------
def save_log(eval_results: Dict[str, float], today_date: str, num_exper: str,
             current_rounds: int, args, client_id: int | None = None) -> None:
    today_dir = os.path.join(args.log_dir, today_date)
    os.makedirs(today_dir, exist_ok=True)

    dataset_dir = os.path.join(today_dir, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    method_name_dir = os.path.join(dataset_dir, args.fl_method)
    os.makedirs(method_name_dir, exist_ok=True)

    log_dir = os.path.join(method_name_dir, num_exper)
    os.makedirs(log_dir, exist_ok=True)

    if client_id is not None:
        client_dir = os.path.join(log_dir, "client_result")
        log_dir = os.path.join(client_dir, f"Client_{client_id}")
    else:
        log_dir = os.path.join(log_dir, "global_result")
    os.makedirs(log_dir, exist_ok=True)

    for metric, value in eval_results.items():
        file_path = os.path.join(log_dir, f"{metric}.csv")
        with open(file_path, 'a') as file:
            file.write(f"{current_rounds},{value}\n")


def get_experiment_num(today_date: str, args) -> str:
    method_name_dir = os.path.join(args.log_dir, today_date, args.dataset_name, args.fl_method)

    if not os.path.exists(method_name_dir):
        return "1"

    exp_dirs = [
        d for d in os.listdir(method_name_dir)
        if os.path.isdir(os.path.join(method_name_dir, d)) and d.isdigit()
    ]
    if not exp_dirs:
        return "1"

    exp_numbers = [int(d) for d in exp_dirs]
    next_exp_num = max(exp_numbers) + 1
    return str(next_exp_num)