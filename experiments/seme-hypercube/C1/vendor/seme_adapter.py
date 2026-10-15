#!/usr/bin/env python3
"""义素超立方体 -> 原仓库 path-star 管线适配器。

把 build_seme_hypercube 的图包装成原仓库 graph dict 形制
（root / paths / edges_forward / edges_reverse），使
build_dataset / MixedBatchSampler / train / eval_hardest_token_acc
原样接驳（prereg.md §2：同架构、同训练循环、同诊断）。

paths = 原点出发的典范测地线：按维序 0..D-1 逐维变值到目标
节点，目标值为 0 的维跳过。路径长 2..D+1 不等 —— 超立方体
叶节点到原点距离本就不等（1..4 跳），是拓扑性质。测试块
（test_dim 维 == test_value）只作路径终点出现：其路径中间
节点 test_dim 恒为 0（test_dim 最后才改），整维隔离。
边例子覆盖全图（含测试区）：模型只有局部边关系、全局路径未见。

原仓库 _build_graph_node_index 假设 paths 划分节点（星图无共
享中间节点）；超立方体共享中间节点，故几何度量附超立方体感
知版（*_h）：节点索引取全部节点各一次。run_smoke.py 在 B0
基线上对原实现做 selftest 对表后方可采信。
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------- 图形制

def seme_node_names(graph: dict) -> list:
    """全部节点名（product 序，节点 id 即下标）。"""
    return ["s_" + "".join(str(int(x)) for x in seme)
            for seme in graph["node_semes"]]


def _weights(n_semes: int, n_values: int) -> np.ndarray:
    return n_values ** np.arange(n_semes)[::-1]


def canonical_path_ids(graph: dict, target: int) -> list:
    """原点 -> target 的典范测地线（节点 id 列表，含两端）。"""
    seme = graph["node_semes"][target]
    w = _weights(graph["n_semes"], graph["n_values"])
    cur = np.zeros(graph["n_semes"], dtype=np.int64)
    ids = [0]
    for d in range(graph["n_semes"]):
        if int(seme[d]) == 0:
            continue
        cur = cur.copy()
        cur[d] = int(seme[d])
        ids.append(int(np.dot(cur, w)))
    return ids


def to_orig_graph(graph: dict) -> dict:
    """原仓库 graph dict 形制。"""
    names = seme_node_names(graph)
    paths = [[names[i] for i in canonical_path_ids(graph, t)]
             for t in range(1, graph["n_nodes"])]
    ef, er = [], []
    for i in range(graph["n_nodes"]):
        for j in graph["adj"][i]:
            if i < j:
                ef.append((names[i], names[j]))
                er.append((names[j], names[i]))
    return {"root": names[0], "paths": paths,
            "edges_forward": ef, "edges_reverse": er}


def build_vocab(orig_graph: dict, node_names: list) -> dict:
    """原仓库 build_vocab 形制：特殊 token 在前（[PAD]=0 …）。"""
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[EDGE]", "[PATH]"]
    all_tokens = special_tokens + node_names
    token2id = {t: i for i, t in enumerate(all_tokens)}
    id2token = {i: t for t, i in token2id.items()}
    return {
        "token2id": token2id, "id2token": id2token,
        "vocab_size": len(all_tokens),
        "pad_id": token2id["[PAD]"], "bos_id": token2id["[BOS]"],
        "eos_id": token2id["[EOS]"], "sep_id": token2id["[SEP]"],
        "edge_id": token2id["[EDGE]"], "path_id": token2id["[PATH]"],
        "root_id": token2id[orig_graph["root"]],
    }


# ------------------------------------------------- 超立方体感知几何度量

def node_index(node_tokens: list, vocab: dict):
    """(node_to_idx, vocab_ids) —— 给定节点名列表各取一次。"""
    node_to_idx = {t: i for i, t in enumerate(node_tokens)}
    vocab_ids = [vocab["token2id"][t] for t in node_tokens]
    return node_to_idx, vocab_ids


def adjacency_h(orig_graph: dict, node_to_idx: dict) -> sp.csr_matrix:
    """稀疏邻接（双向），同 metrics._build_adjacency。"""
    n = len(node_to_idx)
    rows, cols = [], []
    for u, v in orig_graph["edges_forward"] + orig_graph["edges_reverse"]:
        rows.append(node_to_idx[u])
        cols.append(node_to_idx[v])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def eigenspace_gap_h(model, node_tokens, orig_graph, vocab, k: int = 64) -> dict:
    """嵌入在邻接矩阵 top-k 特征向量上的投影能耗（同 compute_eigenspace_gap）。"""
    node_to_idx, vocab_ids = node_index(node_tokens, vocab)
    A = adjacency_h(orig_graph, node_to_idx)
    n_nodes = A.shape[0]
    k = min(k, n_nodes - 2)
    eigenvalues, V_k = spla.eigsh(A.astype(np.float64), k=k, which="LM")
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    V_k = V_k[:, order]
    E = model.emb.weight.detach().cpu().float().numpy()[vocab_ids]
    E_c = E - E.mean(axis=0)
    total_energy = float((E_c ** 2).sum()) + 1e-12
    P = V_k @ (V_k.T @ E_c)
    gap = float(np.linalg.norm(E_c - P, "fro") ** 2 / total_energy)
    per_eigvec_energy = ((V_k.T @ E_c) ** 2).sum(axis=1) / total_energy
    return {"gap": gap, "captured_energy": 1.0 - gap, "k": k,
            "n_nodes": n_nodes, "eigenvalues": eigenvalues.tolist(),
            "per_eigvec_energy": per_eigvec_energy.tolist()}


def stable_rank_h(model, node_tokens, orig_graph, vocab) -> dict:
    """有效维数 ||E||_F^2/||E||_2^2（同 compute_stable_rank）。"""
    _, vocab_ids = node_index(node_tokens, vocab)
    E = model.emb.weight.detach().cpu().float().numpy()[vocab_ids]
    E_c = E - E.mean(axis=0)
    sv = np.linalg.svd(E_c, compute_uv=False)
    frob_sq = float((sv ** 2).sum())
    spec_sq = float(sv[0] ** 2)
    return {"stable_rank": frob_sq / (spec_sq + 1e-30), "singular_values": sv.tolist()}


def principal_angles_h(model, node_tokens, orig_graph, vocab, k: int = 64) -> dict:
    """嵌入子空间与邻接 Laplacian 特征空间的主角度（同 compute_principal_angles）。"""
    node_to_idx, vocab_ids = node_index(node_tokens, vocab)
    A = adjacency_h(orig_graph, node_to_idx)
    n_nodes = A.shape[0]
    k = min(k, n_nodes - 2)
    _, V_k = spla.eigsh(A.astype(np.float64), k=k, which="LM")
    E = model.emb.weight.detach().cpu().float().numpy()[vocab_ids]
    E_c = E - E.mean(axis=0)
    Q, _ = np.linalg.qr(E_c, mode="reduced")
    cosines = np.clip(np.linalg.svd(V_k.T @ Q, compute_uv=False), 0.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    return {"min_angle_deg": float(angles.min()),
            "mean_angle_deg": float(angles.mean()),
            "angles_deg": angles.tolist()}


def arm_cluster_acc_h(model, orig_graph: dict, vocab: dict, device) -> float:
    """最近质心路径分类（原 eval_embedding_auc 原样平移）。

    注意：超立方体共享中间节点，同一节点可属多条路径；
    此数与 B0 的 arm_cluster_acc 只作机械类比，不作等量判读。
    随机基线 = 1/n_paths。
    """
    emb = model.emb.weight.detach().cpu().float().numpy()
    n_arms = len(orig_graph["paths"])
    node_ids, arm_labels = [], []
    for arm_idx, path in enumerate(orig_graph["paths"]):
        for tok in path[1:]:
            node_ids.append(vocab["token2id"][tok])
            arm_labels.append(arm_idx)
    E = emb[node_ids]
    labels = np.array(arm_labels)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_norm = E / np.where(norms > 1e-12, norms, 1.0)
    centroids = np.stack([E_norm[labels == i].mean(axis=0) for i in range(n_arms)])
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(c_norms > 1e-12, c_norms, 1.0)
    sims = E_norm @ centroids.T
    preds = sims.argmax(axis=1)
    return float((preds == labels).mean())


def hop_level_cluster_acc(model, orig_graph: dict, vocab: dict,
                          seme_graph: dict) -> float:
    """按到原点的跳数（seme 空间：非零义素个数）分组的最近质心
    分类准确率 —— 星图 arm 结构的超立方体干净类比。
    分组数 = D+1（跳数 0..D），随机基线 = 1/(D+1)。"""
    emb = model.emb.weight.detach().cpu().float().numpy()
    hop = (seme_graph["node_semes"] != 0).sum(axis=1)
    n_groups = int(hop.max()) + 1
    ids, labels = [], []
    for g in range(n_groups):
        for nid in np.nonzero(hop == g)[0]:
            ids.append(vocab["token2id"]["s_" + "".join(
                str(int(x)) for x in seme_graph["node_semes"][nid])])
            labels.append(g)
    E = emb[ids]
    labels = np.array(labels)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_norm = E / np.where(norms > 1e-12, norms, 1.0)
    centroids = np.stack([E_norm[labels == g].mean(axis=0) for g in range(n_groups)])
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(c_norms > 1e-12, c_norms, 1.0)
    sims = E_norm @ centroids.T
    preds = sims.argmax(axis=1)
    return float((preds == labels).mean())


def alignment_delta_per_hop(model, vocab: dict, graph: dict) -> list:
    """eval_alignment_delta 的逐 hop 分解（06 notebook 的 per-hop
    breakdown）。变长路径：hop k 只统计长度达 k+1 的路径。
    超立方体上 hop-k = 第 k 个被迁移义素的取值迁移。"""
    emb = model.emb.weight.detach().cpu().numpy()
    max_hops = max(len(p) for p in graph["paths"]) - 1
    per_hop = []
    for k in range(max_hops):
        deltas = []
        for path in graph["paths"]:
            if len(path) < k + 2:
                continue
            deltas.append(
                emb[vocab["token2id"][path[k + 1]]]
                - emb[vocab["token2id"][path[k]]]
            )
        dvk = np.stack(deltas)
        mean_dir = dvk.mean(axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
        norms = np.linalg.norm(dvk, axis=1, keepdims=True) + 1e-12
        per_hop.append(float((dvk / norms @ mean_dir).mean()))
    return per_hop
