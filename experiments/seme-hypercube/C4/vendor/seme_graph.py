#!/usr/bin/env python3
"""义素语义图（Seme Semantic Graph）—— V 进制超立方体（Hamming 图）拓扑。

节点 = 义素组合（D 个义素维度 × 每维 V 个取值），共 V^D 个；
边   = 只差一个义素的两个词义（汉明距离恰为 1，共享 D-1 个义素），
       每个节点度 = D·(V-1)。

接口对齐原仓库 src/graph.py（build_path_star_graph / build_vocab /
encode_edge_example 同形制），诊断与训练循环可原样接驳。

清理自 cora-atlas/external/idea-20261014.md 的骨架，修了四处：
1. 删 `encode_edge_seme` 的 `if False else` 死分支与 `_lookup` 线性扫描
   —— 直接用 seme2token 字典（O(1)）。
2. 特殊 token 碰撞：原骨架用 token 0 作 [EDGE]/[SEP]，但 0 是合法义素
   组合 [0,...,0] —— 特殊 token 排词表外（EDGE = V^D, SEP = V^D+1）。
3. 邻接构造 O(n²·D) → O(n·D·V)（逐维变值生成邻居），天然无自环/重边。
4. 接口对齐原仓库：返回 dict 同 build_path_star_graph 形制。

装置自检（确定性）：main() 写 report_structure.json（gates.json 第一批
门限的判据来源）。
"""
import json
from itertools import product
from pathlib import Path

import numpy as np


def build_seme_hypercube(
    n_semes: int = 4,
    n_values: int = 4,
    test_dim: int = None,
    test_value: int = None,
) -> dict:
    """构造义素超立方体图。

    节点：V^D 个，编号 i 的义素向量 = product 序第 i 个组合
          （义素向量当 V 进制数解码即节点编号）。
    边：两个节点相邻 ⇔ 汉明距离恰好为 1。
    划分：第 test_dim 维 == test_value 的整块划为测试
          （从未在任何训练路径中出现，整维隔离）。
    """
    if test_dim is None:
        test_dim = n_semes - 1
        test_value = n_values - 1

    node_semes = np.array(
        list(product(range(n_values), repeat=n_semes)), dtype=np.int64
    )
    n_nodes = len(node_semes)

    # 邻接：逐维变值生成邻居 —— O(n·D·V)
    # product 序下标 = 义素向量当 V 进制数（首位权重最高）
    weights = n_values ** np.arange(n_semes)[::-1]
    adj = {i: [] for i in range(n_nodes)}
    for i, seme in enumerate(node_semes):
        for d in range(n_semes):
            for v in range(n_values):
                if v == seme[d]:
                    continue
                neighbor = seme.copy()
                neighbor[d] = v
                j = int(np.dot(neighbor, weights))
                adj[i].append(j)

    train_nodes = [i for i, s in enumerate(node_semes) if s[test_dim] != test_value]
    test_nodes = [i for i, s in enumerate(node_semes) if s[test_dim] == test_value]

    return {
        "n_semes": n_semes,
        "n_values": n_values,
        "n_nodes": n_nodes,
        "node_semes": node_semes,
        "adj": adj,
        "train_nodes": train_nodes,
        "test_nodes": test_nodes,
        "test_dim": test_dim,
        "test_value": test_value,
    }


def build_vocab(n_semes: int, n_values: int) -> dict:
    """词表：每个义素组合 = 一个 token（id = 超立方体节点编号）。

    特殊 token 排词表外：EDGE = V^D, SEP = V^D + 1
    （token 0 是合法义素组合 [0,...,0]，不可占用 —— 修自原骨架）。
    """
    node_semes = list(product(range(n_values), repeat=n_semes))
    n_nodes = len(node_semes)
    return {
        "token2seme": {i: s for i, s in enumerate(node_semes)},
        "seme2token": {tuple(s): i for i, s in enumerate(node_semes)},
        "edge_token": n_nodes,
        "sep_token": n_nodes + 1,
        "n_tokens": n_nodes + 2,
    }


def encode_edge_example(u, v, vocab: dict) -> list[int]:
    """把一条边编码成训练样本，对齐原仓库 encode_edge_example。

    格式：[EDGE] prev current [SEP]（方向上下文，对齐原文 Phase 1.8b
    接口设计 —— 这样后续可直接验证「接口瓶颈在语义图下是否还成立」）。
    u, v 为义素向量（np 数组或元组）。
    """
    ta = vocab["seme2token"][tuple(int(x) for x in u)]
    tb = vocab["seme2token"][tuple(int(x) for x in v)]
    return [vocab["edge_token"], ta, tb, vocab["sep_token"]]


def structural_metrics(graph: dict) -> dict:
    """装置不变量 —— gates.json 第一批门限的判据来源（确定性）。

    顺带断言：无自环、邻接对称、每条边汉明距离恰为 1。
    """
    n = graph["n_nodes"]
    adj, semes = graph["adj"], graph["node_semes"]
    degrees = [len(adj[i]) for i in range(n)]

    total_edges = 0
    hamming_ok = 0
    for i in range(n):
        assert i not in adj[i], f"自环：节点 {i}"
        for j in adj[i]:
            assert i in adj[j], f"邻接不对称：{i} -> {j}"
            total_edges += 1
            if int(np.count_nonzero(semes[i] != semes[j])) == 1:
                hamming_ok += 1

    train, test = set(graph["train_nodes"]), set(graph["test_nodes"])
    return {
        "n_nodes": n,
        "n_edges": total_edges // 2,
        "node_degree_mean": sum(degrees) / n,
        "node_degree_min": min(degrees),
        "hamming1_edge_fraction": hamming_ok / total_edges,
        "train_test_overlap": len(train & test),
        "coverage": len(train | test) / n,
        "test_block_size": len(test),
    }


def main():
    graph = build_seme_hypercube()
    report = {
        "bench": "seme-hypercube",
        "apparatus": (
            "seme_graph.py build_seme_hypercube(D=4, V=4, "
            "test_dim=3, test_value=3)"
        ),
        "invariants": structural_metrics(graph),
    }
    out = Path(__file__).parent / "report_structure.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
