import torch
from torch import Tensor

class TorchKMeans:
    def __init__(self, n_clusters=8, n_init=1, max_iter=300, tol=5e-3,
                 batch_size=None, is_sparse=False, use_minibatch=False,
                 dtype=None, seed=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.is_sparse = is_sparse
        self.use_minibatch = use_minibatch
        self.dtype = dtype
        self.seed = seed

        self.centroids: Tensor | None = None
        self.labels_: Tensor | None = None

    @torch.no_grad()
    def fit(self, X: Tensor):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # 预处理
        if self.dtype is not None:
            X = X.to(self.dtype)
        X = X - X.mean(dim=0, keepdim=True)
        X = X.contiguous()
        device = X.device
        N, D = X.shape
        K = self.n_clusters

        best_loss = float('inf')
        best_centroids = None

        for _ in range(self.n_init):
            centroids = self._initialize_centroids_kpp(X)

            # Minibatch 累计器（若启用）
            if self.use_minibatch and self.batch_size:
                global_sums = torch.zeros_like(centroids)
                global_counts = torch.zeros(K, device=device, dtype=torch.long)

            prev_loss = float('inf')

            for _ in range(self.max_iter):
                if self.batch_size:
                    idx = torch.randint(0, N, (self.batch_size,), device=device)
                    x = X[idx]
                else:
                    x = X

                # 计算到各中心的平方距离 (B, K)
                dist2 = _squared_euclidean(x, centroids)

                # 归属与损失
                labels = torch.argmin(dist2, dim=1)
                loss = dist2.gather(1, labels.view(-1, 1)).sum()

                # 更新中心
                if self.use_minibatch and self.batch_size:
                    # 累计到全局
                    sums = torch.zeros_like(centroids)
                    sums.index_add_(0, labels, x)
                    counts = torch.bincount(labels, minlength=K)
                    global_sums += sums
                    global_counts += counts
                    new_centroids = global_sums / global_counts.clamp_min(1).unsqueeze(1)
                else:
                    sums = torch.zeros_like(centroids)
                    sums.index_add_(0, labels, x)
                    counts = torch.bincount(labels, minlength=K)
                    # 空簇重置为随机样本（避免 NaN）
                    empty = counts == 0
                    if empty.any():
                        rand_idx = torch.randint(0, x.size(0), (empty.sum().item(),), device=device)
                        sums[empty] = x[rand_idx]
                        counts[empty] = 1
                    new_centroids = sums / counts.unsqueeze(1)

                # is_sparse 的“零中心”兼容（如果需要保留）
                if self.is_sparse:
                    new_centroids[0].zero_()

                centroids = new_centroids

                # 早停（相对改变量，避免 inf/0）
                rel = torch.abs(prev_loss - loss) / (prev_loss + 1e-12)
                prev_loss = loss
                if rel < self.tol:
                    break

            # 记录最优
            if loss < best_loss:
                best_loss = loss
                best_centroids = centroids

        # 最终全量打标
        self.centroids = best_centroids
        all_dist2 = _squared_euclidean(X, self.centroids)
        self.labels_ = torch.argmin(all_dist2, dim=1)
        return self

    @torch.no_grad()
    def _initialize_centroids_kpp(self, X: Tensor, cap: int = 4_194_304):
        N = X.size(0)
        device = X.device
        K = self.n_clusters

        # 子采样（同你原逻辑）
        if N > cap:
            idx = torch.randperm(N, device=device)[:cap]
            Xs = X[idx]
        else:
            Xs = X

        # 起点：稀疏模式则用零向量，否则随机
        centroids = []
        if self.is_sparse:
            c0 = torch.zeros_like(Xs[0])
        else:
            c0 = Xs[torch.randint(0, Xs.size(0), (1,), device=device)].squeeze(0)
        centroids.append(c0)

        # 维护“到最近中心的最小平方距离”
        d2_min = _squared_euclidean(Xs, c0.unsqueeze(0)).squeeze(1)  # (Ns,)

        for _ in range(1, K):
            # 概率 ∝ d^2
            probs = d2_min.clamp_min(1e-12)
            probs = probs / probs.sum()
            next_idx = torch.multinomial(probs, 1)
            c_new = Xs[next_idx].squeeze(0)
            centroids.append(c_new)
            # 仅与新中心比较并更新最小距离
            new_d2 = _squared_euclidean(Xs, c_new.unsqueeze(0)).squeeze(1)
            d2_min = torch.minimum(d2_min, new_d2)

        return torch.stack(centroids, dim=0).to(device)


@torch.no_grad()
def _squared_euclidean(a: Tensor, b: Tensor) -> Tensor:
    """
    a: (B, D), b: (K, D) -> returns (B, K) squared distances
    """
    a2 = (a * a).sum(dim=1, keepdim=True)        # (B, 1)
    b2 = (b * b).sum(dim=1, keepdim=True).t()    # (1, K)
    # 使用 GEMM；对大维度通常比 cdist 快且不需要 sqrt
    ab = a @ b.t()                                # (B, K)
    dist2 = a2 - 2 * ab + b2
    # 数值上可能出现极小负数，截断为非负
    return dist2.clamp_min_(0.0)
