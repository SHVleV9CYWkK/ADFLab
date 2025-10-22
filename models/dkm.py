import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTeacherDKMLayer(nn.Module):
    """
    Differentiable k-Means with multi-teacher soft alignment
    — 每次 forward 内部 E/M；质心非 Parameter
    — 语义 (Jaccard) × 数值 (exp-dist) 混合
    """

    def __init__(
        self,
        n_clusters: int = 16,
        max_iter: int = 10,
        tau: float = 1e-3,
        eps: float = 1e-4,
        alpha_mix: float = 0.7,     # 语义占比 α
        beta_dist: float = 1.0,     # 数值距离温度 β
        beta_sem: float = 5.0,      # 教师软索引篇温度 β
    ):
        super().__init__()
        self.K = n_clusters
        self.T_max = max_iter
        self.tau = tau
        self.eps = eps
        self.alpha_mix = alpha_mix
        self.beta_dist = beta_dist
        self.beta_sem = beta_sem
        self.C = None

    def forward(
            self,
            X: torch.Tensor,  # (N, D)
            *,
            teacher_centroids: torch.Tensor | None = None,  # (T, K, D)
            teacher_alphas: torch.Tensor | None = None,  # (T,)
            teacher_index_tables: list[torch.Tensor] | None = None,  # 兼容参数（此实现使用基于 X 的 soft 语义）
            lambda_teacher: float = 1.0,
    ):
        """
        返回:
            X_rec : (N, D)  # 与输入 X 同一坐标系（若内部做了中心化，这里已加回均值）
            C_out : (K, D)  # 学生质心，亦在输入坐标系
            A     : (N, K)  # 学生软分配
        """
        device = X.device
        N, D = X.shape

        # ---- 统一坐标系：是否中心化（默认 True，可在 __init__ 里设置 self.centering） ----
        mu = X.mean(dim=0, keepdim=True)  # (1, D)
        Xc = X - mu  # 在中心化空间里聚类

        # ---- 质心初始化（在当前坐标系） ----
        with torch.no_grad():  # 初始化/取旧值不需要梯度
            if self.C is None:
                C0 = self._kpp_init(X)  # k++ 初始化
            else:
                C0 = self.C

        Cc = C0.detach().clone()  # 确保是 leaf，无历史图

        # ---- 教师准备（平移到同坐标；alphas 归一化） ----
        use_teacher = (teacher_centroids is not None) and (lambda_teacher > 0.0)
        if use_teacher:
            Tc = teacher_centroids.to(device)  # (T, K, D)
            Tc = Tc - (mu if mu is not None else 0.0)  # 保证与 Xc 同坐标
            T = Tc.size(0)

            if teacher_alphas is None:
                alphas = torch.full((T,), 1.0 / T, device=device)
            else:
                alphas = teacher_alphas.to(device)
                alphas = alphas / (alphas.sum() + 1e-8)

            # 说明：本实现对“语义相似”统一采用基于 X 的 soft 方式（teacher_index_tables 保留但不使用），
            # 可避免不同数据集 N_t != N 的维度不一致问题。
        else:
            lambda_teacher = 0.0
            T = 0
            alphas = None
            Tc = None

        # ---- E/M 迭代 ----
        for _ in range(self.T_max):
            # E-step：学生软分配（自适应温度，防止饱和）
            dist2 = torch.cdist(Xc, Cc, p=2).pow(2)  # (N, K)
            tau_eff = self.tau if (getattr(self, "tau", 0.0) and self.tau > 0) else 1.0
            scale_s = dist2.median().detach() + 1e-8  # 距离尺度自适应
            logits = - dist2 / (scale_s * tau_eff)
            A = torch.softmax(logits, dim=1)  # (N, K)

            # 教师软对齐：语义×数值混合得到映射权重 w，并汇总教师拉动项
            if use_teacher:
                # (1) 语义相似：在同一 X 上评估 teacher soft assignment（自适应温度）
                teacher_soft = []
                for t in range(T):
                    dist_xt = torch.cdist(Xc, Tc[t], p=2)  # (N, K)
                    scale_t = dist_xt.median().detach() + 1e-8
                    prob_t = torch.softmax(- self.beta_sem * dist_xt / scale_t, dim=1)
                    teacher_soft.append(prob_t)  # (N, K)

                # soft-Jaccard：每个老师与学生在 K×K 的“簇-簇重叠”
                jac_list = []
                for t in range(T):
                    inter = teacher_soft[t].T @ A  # (K, K)
                    union = teacher_soft[t].sum(0, keepdim=True).T + A.sum(0, keepdim=True) - inter + 1e-8
                    jac_list.append(inter / union)
                J = torch.stack(jac_list, dim=0)  # (T, K, K)

                # (2) 数值相似：教师质心 ↔ 学生质心（自适应缩放）
                dist_c = torch.cdist(Tc, Cc, p=2)  # (T, K, K)
                scale_c = dist_c.median().detach() + 1e-8
                S = torch.exp(- self.beta_dist * dist_c / scale_c)  # (T, K, K)

                # (3) 语义×数值幂次混合，并沿 j 维（学生簇）行归一化得到映射 w_{t,i->j}
                M = (J + 1e-8).pow(self.alpha_mix) * (S + 1e-8).pow(1.0 - self.alpha_mix)  # (T, K, K)
                w = M / (M.sum(dim=2, keepdim=True) + 1e-8)  # (T, K, K)

                # (4) 教师拉动：teacher_matched[j, d] = Σ_t α_t Σ_i w[t, i, j] * Tc[t, i, d]
                teacher_matched = torch.einsum('t,tik,tid->kd', alphas, w, Tc)  # (K, D)
            else:
                teacher_matched = 0.0

            # M-step：带教师先验的质心更新
            num_stu = A.sum(dim=0, keepdim=True).t()  # (K, 1)
            Cc_new = (A.t() @ Xc + lambda_teacher * teacher_matched) / (num_stu + lambda_teacher + 1e-8)  # (K, D)

            # 收敛判定：相对范数，避免尺度问题
            num = torch.norm(Cc_new - Cc)
            den = torch.norm(Cc) + 1e-8
            if (num / den) < self.eps:
                Cc = Cc_new
                break
            Cc = Cc_new

        # ---- 重构并还原到原坐标系（若中心化） ----
        Xc_rec = A @ Cc
        X_rec = Xc_rec + mu  # (N, D)
        C_out = Cc + mu  # (K, D)

        # 内部缓存质心（保留当前内部坐标版本；若想对外一致，也可改存 C_out）
        self.C = Cc.detach()
        return X_rec, C_out, A

    def _kpp_init(self, X):
        """k-means++ 初始化, 返回 (K,D)"""
        with torch.no_grad():
            N, _ = X.shape
            device = X.device
            idx = torch.randint(0, N, (1,), device=device)
            C = [X[idx].squeeze(0)]
            for _ in range(1, self.K):
                D2 = torch.cdist(X, torch.stack(C), p=2).pow(2).min(dim=1)[0]
                probs = D2 / (D2.sum() + 1e-8)
                nxt = torch.multinomial(probs, 1)
                C.append(X[nxt].squeeze(0))
            return torch.stack(C)
