from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterable

import numpy as np
import torch
import torcheval.metrics.functional as metrics
from torch.utils.data import DataLoader, Subset
from utils.utils import get_optimizer, get_lr_scheduler


class Client(ABC):
    """
    基类约定（与异步协调器配合）：
    - 协调器在客户端 Join 时调用：
        set_init_model(...) -> init_client()
      之后用 compute_time_for_next_burst() 计算首个 TRAIN_DONE 的持续时间并排程。
    - 每次 TRAIN_DONE 触发时：
        client.train() -> client.on_train_done(now)
        payload = client.send_model()  # CPU state + meta
        协调器按“在线过滤 + 少推”把 payload 发给若干邻居，邻居执行 receive_neighbor_model(payload)
    - 若 fuse_on_receive=True，则接收即融合；否则缓冲到 neighbor_model_weights，等待子类在合适时机调用 aggregate()

    子类需要实现：
        - train(self)
        - aggregate(self)
        - set_init_model(self, model)
    可以选择直接调用本类的 _local_train() 与 _weight_aggregation() 来复用默认训练与平均逻辑。
    """

    # -----------------------------
    # 初始化 / 资源占用延迟到 init_client()
    # -----------------------------
    def __init__(self, client_id: int, dataset_index: Dict[str, str],
                 full_dataset, hyperparam: Dict[str, Any], device: torch.device):
        self.id = client_id
        self.device = device

        # ======= 模型 & 优化相关 =======
        self.model: Optional[torch.nn.Module] = None
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.optimizer_name = hyperparam.get('optimizer_name', 'sgd')
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr = float(hyperparam.get('lr', 0.1))
        self.epochs = int(hyperparam.get('local_epochs', 1))
        self.scheduler_name = hyperparam.get('scheduler_name', 'none')
        self.n_rounds = int(hyperparam.get('n_rounds', 0))
        self.lr_scheduler = None
        self.last_accuracy: Optional[torch.Tensor] = None

        # ======= 数据集索引（延迟构建 DataLoader）=======
        train_indices = np.load(dataset_index['train']).tolist()
        val_indices = np.load(dataset_index['val']).tolist()
        self._full_dataset = full_dataset
        self._train_indices = train_indices
        self._val_indices = val_indices

        self.train_dataset_len = len(train_indices)  # n_i
        self.val_dataset_len = len(val_indices)
        self.num_classes = len(full_dataset.classes)
        self.batch_size = int(hyperparam.get('bz', 32))  # B_i

        # DataLoader 延迟到 init_client() 再创建（Join 时才占资源）
        self.client_train_loader: Optional[DataLoader] = None
        self.client_val_loader: Optional[DataLoader] = None

        # ======= 异步元信息（时间轴/计数/通信）=======
        self.local_version: int = 0                 # 每完成一次本地训练 +1
        self.last_update_time: float = 0.0
        self.steps_seen: int = 0
        self.samples_seen: int = 0
        self.bytes_sent: int = 0                    # 由协调器按实际收件人数量更新
        self.bytes_recv: int = 0

        # 训练时钟配置（确定性、可复现）
        # 支持两种模式：'constant' 或 'steps*t_step'（默认）
        self.compute_time_mode: str = hyperparam.get('compute_time_mode', 'steps*t_step')
        self.compute_interval: float = float(hyperparam.get('compute_interval', 1.0))  # constant 模式使用
        self.t_step: float = float(hyperparam.get('t_step', 0.01))  # 每步耗时，steps*t_step 模式使用

        # ======= 接收端策略（异步下的缓冲与即时融合）=======
        self.fuse_on_receive: bool = bool(hyperparam.get('fuse_on_receive', True))
        self.buffer_limit: int = int(hyperparam.get('buffer_limit', 10))
        self.neighbor_model_weights: list[Dict[str, torch.Tensor]] = []

        # 评估时的全局指标记录（保留原字段）
        self.global_metric = self.global_epoch = 0

    # -----------------------------
    # 资源初始化（在 Join 时调用）
    # -----------------------------
    def init_client(self):
        """在 Join 发生时初始化 DataLoader / Optimizer / LR Scheduler。"""
        # DataLoader
        client_train_dataset = Subset(self._full_dataset, indices=self._train_indices)
        client_val_dataset = Subset(self._full_dataset, indices=self._val_indices)
        self.client_train_loader = DataLoader(
            client_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True  # 与原实现保持一致
        )
        effective_val_bz = self.batch_size if self.batch_size <= len(client_val_dataset) else len(client_val_dataset)
        self.client_val_loader = DataLoader(
            client_val_dataset,
            batch_size=effective_val_bz,
            shuffle=False
        )

        # Optimizer & Scheduler
        if self.model is None:
            raise RuntimeError("init_client() 之前必须先调用 set_init_model(model)")
        self.optimizer = get_optimizer(self.optimizer_name, self.model.parameters(), self.lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.scheduler_name, self.n_rounds)

    # -----------------------------
    # 异步：训练时钟 & 事件元信息
    # -----------------------------
    def compute_time_for_next_burst(self) -> float:
        """
        返回“下一次本地训练单元（burst）”所需的模拟墙钟时间。
        - constant: 直接返回 compute_interval
        - steps*t_step: 返回 steps_per_burst * t_step
        """
        if self.compute_time_mode == 'constant':
            return float(self.compute_interval)
        # 需要 DataLoader 已初始化（在 Join 后）
        steps = self._steps_per_burst()
        return float(steps) * float(self.t_step)

    def on_train_done(self, now: float) -> None:
        """
        由协调器在本地训练完成的时刻调用，更新异步元信息（版本/时间/计数）。
        """
        self.local_version += 1
        self.last_update_time = float(now)

        steps = self._steps_per_burst()
        self.steps_seen += steps
        # 以 batch_size × steps 估计处理样本数（与 drop_last=True 一致、可复现）
        self.samples_seen += steps * self.batch_size

    # -----------------------------
    # 默认训练实现（子类可直接调用）
    # -----------------------------
    def _local_train(self):
        """默认本地训练：跑 self.epochs 个 epoch。子类的 train() 可直接调用它。"""
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader 尚未初始化：请先在 Join 时调用 init_client()。")
        self.model.train()
        for _ in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()

    # -----------------------------
    # 发送 / 接收（载荷规范）
    # -----------------------------
    def send_model(self) -> Dict[str, Any]:
        """
        返回“可传输载荷”：CPU 上的 state_dict + 元信息（sender_id / version / sender_time / model_nbytes）。
        注意：bytes_sent/recv 的累计应由协调器按收件人数量统计，这里仅提供 model_nbytes 便于计费。
        """
        if self.model is None:
            raise RuntimeError("send_model() 之前必须先设置模型：set_init_model(model)")
        state_cpu = {k: v.detach().clone().to('cpu') for k, v in self.model.state_dict().items()}
        payload = {
            "state": state_cpu,
            "meta": {
                "sender_id": self.id,
                "version": self.local_version,
                "sender_time": self.last_update_time,
                "model_nbytes": self._model_num_bytes(),
            }
        }
        return payload

    def receive_neighbor_model(self, neighbor_model: Dict[str, Any] | Dict[str, torch.Tensor]):
        """
        接收邻居的模型载荷。兼容两类输入：
          - {'state': state_dict_on_cpu, 'meta': {...}}
          - 直接的 state_dict_on_cpu（向后兼容旧调用）
        策略：
          - 若 fuse_on_receive=True：收到即融合（调用子类实现的 aggregate()），并清空缓冲。
          - 否则：入缓冲；若超过 buffer_limit，丢弃最旧项。
        """
        # 提取 state_dict（CPU）
        if isinstance(neighbor_model, dict) and "state" in neighbor_model:
            state_dict = neighbor_model["state"]
        else:
            state_dict = neighbor_model  # 兼容旧调用：直接就是 state_dict

        # 基本防御
        if not isinstance(state_dict, dict):
            raise ValueError("receive_neighbor_model() 期望收到 state_dict 或 {'state': state_dict, ...}。")

        self.neighbor_model_weights.append(state_dict)

        # 控制缓冲大小
        if self.buffer_limit is not None and self.buffer_limit > 0:
            overflow = len(self.neighbor_model_weights) - self.buffer_limit
            if overflow > 0:
                # 丢弃最旧
                self.neighbor_model_weights = self.neighbor_model_weights[overflow:]

        # 收到即融合（推荐默认）
        if self.fuse_on_receive:
            # 由子类在 aggregate() 内部将 neighbor_model_weights 聚合到 self.model
            self.aggregate()
            # 清空缓冲
            self.neighbor_model_weights.clear()

    # -----------------------------
    # 评估与 LR 调度（沿用原实现）
    # -----------------------------
    def evaluate_model(self) -> Dict[str, float]:
        if self.client_val_loader is None:
            raise RuntimeError("评估需要已初始化的验证 DataLoader：请先在 Join 时调用 init_client()。")

        self.model.eval()
        total_loss = 0.0
        all_labels: list[torch.Tensor] = []
        all_predictions: list[torch.Tensor] = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                total_loss += float(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                all_labels.append(labels)
                all_predictions.append(predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        avg_loss = total_loss / max(1, len(self.client_val_loader))
        accuracy = metrics.multiclass_accuracy(all_predictions, all_labels, num_classes=self.num_classes)
        precision = metrics.multiclass_precision(all_predictions, all_labels, num_classes=self.num_classes)
        recall = metrics.multiclass_recall(all_predictions, all_labels, num_classes=self.num_classes)
        f1 = metrics.multiclass_f1_score(all_predictions, all_labels, average="weighted", num_classes=self.num_classes)

        self.last_accuracy = accuracy
        return {
            'loss': float(avg_loss),
            'accuracy': float(accuracy.cpu().item()),
            'precision': float(precision.cpu().item()),
            'recall': float(recall.cpu().item()),
            'f1': float(f1.cpu().item()),
        }

    def update_lr(self):
        if self.last_accuracy is not None and self.lr_scheduler is not None:
            self.lr_scheduler.step(self.last_accuracy)

    # -----------------------------
    # 聚合与模型设置（由子类实现）
    # -----------------------------
    @abstractmethod
    def train(self):
        """建议子类内部直接调用 self._local_train()，除非需要自定义本地优化过程。"""
        raise NotImplementedError

    @abstractmethod
    def aggregate(self):
        """
        子类应读取 self.neighbor_model_weights（列表，元素为 CPU 上的 state_dict），
        将其聚合到 self.model（通常：to(self.device) 后做平均/加权融合），完成后可不必清空缓冲（本类在调用后已清空）。
        可复用 self._weight_aggregation() 获得简单平均的 state_dict。
        """
        raise NotImplementedError

    @abstractmethod
    def set_init_model(self, model: torch.nn.Module):
        """在 Join 前由协调器调用，设置初始模型（一般为全局初始或预训练），随后 init_client() 才能创建优化器。"""
        raise NotImplementedError

    # -----------------------------
    # 工具：平均、步数、模型大小
    # -----------------------------
    def _weight_aggregation(self) -> Dict[str, torch.Tensor]:
        """对缓冲中的若干 state_dict 做简单平均（在 device 上），返回新的权重字典。"""
        if len(self.neighbor_model_weights) == 0:
            raise RuntimeError("neighbor_model_weights 为空，无法聚合。")
        average_weights: Dict[str, torch.Tensor] = {}
        keys = list(self.neighbor_model_weights[0].keys())
        for k in keys:
            # 累加到 device，再做平均
            acc = None
            for sd in self.neighbor_model_weights:
                tensor = sd[k].to(self.device)
                acc = tensor if acc is None else (acc + tensor)
            average_weights[k] = acc / float(len(self.neighbor_model_weights))
        return average_weights

    def _steps_per_burst(self) -> int:
        """
        每个训练 burst 的步数 = epochs * len(train_loader)。
        注意：train_loader 使用 drop_last=True，与原实现保持一致（步数为 floor(n_i / B) 每 epoch）。
        """
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader 尚未初始化：请先在 Join 时调用 init_client()。")
        return int(self.epochs * len(self.client_train_loader))

    def _model_num_bytes(self) -> int:
        """估算当前模型参数字节数（用于通信会计；不含 buffer/优化器状态）。"""
        if self.model is None:
            return 0
        total = 0
        for p in self.model.state_dict().values():
            total += p.numel() * p.element_size()
        return int(total)