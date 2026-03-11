import torch
import torch.nn.functional as F
from typing import Dict, Optional

from clients.dfl_method_clients.async_fl.pushsum_client import ADFedPushSumClient

class ProtoKDClient(ADFedPushSumClient):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        self.temperature = hyperparam.get('temperature', 0.1)
        self.proto_weight = hyperparam.get('proto_weight', 0.1)  # 对比损失的权重 lambda

        # 存储特征空间的原型
        self.local_prototypes: Dict[int, torch.Tensor] = {}
        self.global_prototypes: Dict[int, torch.Tensor] = {}

        # 用于 Hook 提取特征的临时变量
        self._current_features = None
        self._hook_handle = None

        self._global_classes = []
        self._class_to_idx: Dict[int, int] = {}
        self._proto_matrix: Optional[torch.Tensor] = None


    # ---- 对比损失计算 ----
    def _compute_contrastive_loss(self, features, labels):
        if self._proto_matrix is None or self._proto_matrix.size(0) < 2:
            return features.new_zeros(())

        features = F.normalize(features, p=2, dim=1)
        proto_matrix = F.normalize(self._proto_matrix, p=2, dim=1)

        sim_matrix = features @ proto_matrix.T / self.temperature

        target = torch.full_like(labels, -1)
        for c, idx in self._class_to_idx.items():
            target[labels == c] = idx

        valid = target >= 0
        if not valid.any():
            return features.new_zeros(())

        return F.cross_entropy(sim_matrix[valid], target[valid])

    def _refresh_proto_cache(self):
        if not self.global_prototypes:
            self._proto_matrix = None
            self._class_to_idx = None
            return

        classes = sorted(self.global_prototypes.keys())
        self._global_classes = classes
        self._class_to_idx = {c: i for i, c in enumerate(classes)}
        self._proto_matrix = torch.stack([self.global_prototypes[c] for c in classes], dim=0)

    @torch.no_grad()
    def _aggregate_neighbor_prototypes(self):
        class_proto_list = {}

        for state, meta in self.neighbor_model_weights_buffer:
            neighbor_protos = meta.get("prototypes", None)
            if not neighbor_protos:
                continue

            for label, proto in neighbor_protos.items():
                label = int(label)
                proto = proto.to(self.device, dtype=torch.float32, non_blocking=True)
                proto = F.normalize(proto, p=2, dim=0)

                if label not in class_proto_list:
                    class_proto_list[label] = []
                class_proto_list[label].append(proto)

        new_global_prototypes = {}
        for label, proto_list in class_proto_list.items():
            mean_proto = torch.stack(proto_list, dim=0).mean(dim=0)
            new_global_prototypes[label] = F.normalize(mean_proto, p=2, dim=0)

        self.global_prototypes = new_global_prototypes
        self._refresh_proto_cache()


    def _register_feature_hook(self):
        """
        动态注册 hook，截取分类头前一层的输入作为特征向量。
        无需修改原有的 LeNet/ResNet18 代码结构。
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()

        def hook_fn(module, input):
            # 截取全连接层的输入作为特征 (通常是 flatten 后的 tensor)
            self._current_features = input[0]

        # 根据模型类型挂载 Hook。你需要根据你实际模型中分类头的名字调整
        # ResNet18 通常分类头叫 'fc'，LeNet 通常最后层叫 'fc3' 或 'classifier'
        if hasattr(self.model, 'fc1'):
            self._hook_handle = self.model.fc1.register_forward_pre_hook(hook_fn)
        elif hasattr(self.model, 'fc3'):
            self._hook_handle = self.model.fc3.register_forward_pre_hook(hook_fn)
        else:
            # Fallback: 挂载到最后一个模块
            last_module = list(self.model.named_modules())[-1][1]
            self._hook_handle = last_module.register_forward_pre_hook(hook_fn)


    def set_init_model(self, model):
        """重写初始化，注册 Forward Hook 提取特征"""
        super().set_init_model(model)
        self._register_feature_hook()

    # ---- 覆盖通信逻辑 ----
    def send_model(self):
        """在发送载荷中加入本地原型"""
        state, meta = super().send_model()
        # 这里假设 self.train_loader 是你类里的本地数据加载器
        # 如果不是，请替换为实际获取本地数据的变量
        meta["prototypes"] = self.local_prototypes
        return state, meta

    @torch.no_grad()
    def aggregate(self):
        self._aggregate_neighbor_prototypes()
        super().aggregate()

    # ---- 本地训练流程改写 ----
    def train(self):
        """
        覆盖原有的 train 方法，为 delayed client 注入蒸馏损失
        这里我写了一个标准的本地训练循环骨架，你需要用你的实际 optimizer/dataloader 适配
        """
        if self.client_train_loader is None:
            raise RuntimeError("DataLoader 尚未初始化：请先在 Join 时调用 init_client()。")
        self.model.train()

        proto_sums = {}
        proto_counts = {}

        for _ in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x)

                # 1. 标准交叉熵
                loss_ce = self.criterion(outputs, labels).mean()
                loss = loss_ce

                # 2. 如果是延迟节点，且已经收到了全局原型，则叠加对比损失
                # if self.is_delayed_client and len(self.global_prototypes) > 0:
                if len(self.global_prototypes) > 0:
                    # 此时 self._current_features 已经被 hook 捕获
                    loss_proto = self._compute_contrastive_loss(self._current_features, labels)
                    loss += self.proto_weight * loss_proto

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    features_detached = self._current_features.detach()

                    for c in labels.unique():
                        c_int = int(c.item())
                        mask = labels == c
                        feat_sum = features_detached[mask].sum(dim=0)
                        cnt = int(mask.sum().item())

                        if c_int not in proto_sums:
                            proto_sums[c_int] = feat_sum.clone()
                            proto_counts[c_int] = cnt
                        else:
                            proto_sums[c_int] += feat_sum
                            proto_counts[c_int] += cnt

            # 训练结束后直接由累计量生成本地原型，不再额外跑一遍数据
            self.local_prototypes = {
                c: F.normalize(proto_sums[c] / proto_counts[c], p=2, dim=0).detach()
                for c in proto_sums
                if proto_counts[c] > 0
            }
