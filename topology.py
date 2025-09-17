# topology.py
from __future__ import annotations

from typing import List, Optional
import random


def generate_overlay(num_clients: int,
                     num_conn: int,
                     symmetry: int,
                     seed: int) -> List[List[int]]:
    """
    生成一次性的覆盖图（overlay），返回 0/1 邻接矩阵（list[list[int]]）。
    - symmetry != 0: 生成无向 k-正则图（度≈num_conn），确保连通；
    - symmetry == 0: 生成有向图（每个节点出度=num_conn），无自环；
    采用固定 seed 的伪随机过程，确保可复现。

    约束：
      - 无向：num_conn <= num_clients-1 且 num_clients * num_conn 为偶数；
      - 有向：num_conn < num_clients （禁止自环）。
    """
    n = int(num_clients)
    k = int(num_conn)
    if n <= 0:
        raise ValueError("num_clients must be positive.")

    rng = random.Random(int(seed))
    graph: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]

    if symmetry != 0:
        # -------- 无向：近似 k-正则，保证连通 --------
        if k < 0 or k > n - 1:
            raise ValueError("Undirected: num_conn must be in [0, n-1].")
        if (n * k) % 2 != 0:
            raise ValueError("Undirected: n * num_conn must be even.")

        if k == 0:
            return graph  # 全零

        degree = [0] * n
        nodes = list(range(n))
        rng.shuffle(nodes)

        # 1) 先生成一棵随机生成树（保证连通）
        for i in range(n - 1):
            u, v = nodes[i], nodes[i + 1]
            if graph[u][v] == 0:
                graph[u][v] = graph[v][u] = 1
                degree[u] += 1
                degree[v] += 1

        target_edges = (n * k) // 2
        current_edges = n - 1  # 生成树已有 n-1 条边

        # 2) 随机补边（仅在度未满的节点间补）
        def refresh_available():
            return [i for i in range(n) if degree[i] < k]

        available = refresh_available()
        while current_edges < target_edges and len(available) >= 2:
            u, v = rng.sample(available, 2)
            if u != v and graph[u][v] == 0:
                graph[u][v] = graph[v][u] = 1
                degree[u] += 1
                degree[v] += 1
                current_edges += 1
                available = refresh_available()

        # 3) 确定性补边兜底
        if current_edges < target_edges:
            for u in range(n):
                for v in range(u + 1, n):
                    if degree[u] < k and degree[v] < k and graph[u][v] == 0:
                        graph[u][v] = graph[v][u] = 1
                        degree[u] += 1
                        degree[v] += 1
                        current_edges += 1
                        if current_edges == target_edges:
                            break
                if current_edges == target_edges:
                    break

        return graph

    else:
        # -------- 有向：每节点出度 = k，禁止自环 --------
        if k < 0 or k >= n:
            raise ValueError("Directed: num_conn must be in [0, n-1).")

        if k == 0:
            return graph  # 全零

        outdeg = [0] * n
        nodes = list(range(n))
        rng.shuffle(nodes)

        # 1) 先连一条随机链（促进弱连通）
        for i in range(n - 1):
            u, v = nodes[i], nodes[i + 1]
            if u != v and graph[u][v] == 0:
                graph[u][v] = 1
                outdeg[u] += 1

        # 2) 补齐每个节点的出度至 k
        for u in range(n):
            if outdeg[u] >= k:
                continue
            pool = [v for v in range(n) if v != u and graph[u][v] == 0]
            rng.shuffle(pool)
            need = k - outdeg[u]
            for v in pool[:need]:
                graph[u][v] = 1
                outdeg[u] += 1

        return graph


def out_neighbors(adj: List[List[int]], i: int) -> list[int]:
    """返回节点 i 的出邻居列表（索引）。"""
    row = adj[i]
    return [j for j, x in enumerate(row) if j != i and x]


def validate_connected_undirected(adj: List[List[int]]) -> bool:
    """
    粗略连通性检查（无向）。用于开发/测试阶段，可选。
    """
    n = len(adj)
    if n == 0:
        return True
    # BFS
    seen = [False] * n
    q = [0]
    seen[0] = True
    while q:
        u = q.pop()
        for v in range(n):
            if u != v and (adj[u][v] or adj[v][u]) and not seen[v]:
                seen[v] = True
                q.append(v)
    return all(seen)