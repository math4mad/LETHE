"""Evaluation metrics and geometric diagnostics for the path-star task."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Core task metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_hardest_token_acc(model, vocab: dict, graph: dict, test_leaves: set, device) -> float:
    """Accuracy of predicting the first path hop from a held-out test leaf."""
    model.eval()
    correct = 0
    leaf_to_first_hop = {path[-1]: path[-2] for path in graph["paths"]}
    for leaf in test_leaves:
        input_ids = torch.tensor(
            [[vocab["path_id"], vocab["token2id"][leaf], vocab["sep_id"]]],
            dtype=torch.long, device=device,
        )
        logits = model(input_ids)
        pred_id = logits[0, 2, :].argmax().item()
        correct += int(pred_id == vocab["token2id"][leaf_to_first_hop[leaf]])
    return correct / max(len(test_leaves), 1)


@torch.no_grad()
def eval_embedding_auc(model, vocab: dict, graph: dict, device) -> float:
    """
    Nearest-centroid arm classification accuracy on all arm nodes.
    Also known as arm_cluster_acc in the research plan.
    Random baseline = 1/n_arms.
    """
    emb = model.emb.weight.detach().cpu().float().numpy()
    n_arms = len(graph["paths"])

    node_ids, arm_labels = [], []
    for arm_idx, path in enumerate(graph["paths"]):
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


# arm_cluster_acc is the same metric as eval_embedding_auc
arm_cluster_acc = eval_embedding_auc


@torch.no_grad()
def eval_alignment_delta(model, vocab: dict, graph: dict) -> float:
    """Consistency of hop-to-hop difference vectors across arms (random baseline ≈ 0)."""
    emb = model.emb.weight.detach().cpu().numpy()
    path_length = len(graph["paths"][0]) - 1
    degree = len(graph["paths"])

    diff_vecs = np.zeros((degree, path_length, emb.shape[1]))
    for arm_idx, path in enumerate(graph["paths"]):
        for k in range(1, path_length + 1):
            diff_vecs[arm_idx, k - 1] = (
                emb[vocab["token2id"][path[k]]] - emb[vocab["token2id"][path[k - 1]]]
            )

    cos_sims = []
    for k in range(path_length):
        dvk = diff_vecs[:, k, :]
        mean_dir = dvk.mean(axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
        norms = np.linalg.norm(dvk, axis=1, keepdims=True) + 1e-12
        cos_sims.extend((dvk / norms @ mean_dir).tolist())

    return float(np.mean(cos_sims))


def steps_to_pct(log: list[dict], target_acc: float) -> int | None:
    """
    Interpolate training log to find the step where hardest_token_acc first reaches target_acc.
    Returns None if never reached.
    """
    for i, entry in enumerate(log):
        if entry["hardest_token_acc"] >= target_acc:
            if i == 0:
                return entry["step"]
            prev = log[i - 1]
            cur = entry
            frac = (target_acc - prev["hardest_token_acc"]) / max(
                cur["hardest_token_acc"] - prev["hardest_token_acc"], 1e-12
            )
            return int(prev["step"] + frac * (cur["step"] - prev["step"]))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Graph index helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph_node_index(graph: dict, vocab: dict) -> tuple[list, dict, list]:
    """Returns (node_tokens, node_to_idx, vocab_ids) for all graph nodes."""
    node_tokens = ["v_root"]
    for path in graph["paths"]:
        for tok in path[1:]:
            node_tokens.append(tok)
    node_to_idx = {t: i for i, t in enumerate(node_tokens)}
    vocab_ids = [vocab["token2id"][t] for t in node_tokens]
    return node_tokens, node_to_idx, vocab_ids


def _build_adjacency(graph: dict, node_to_idx: dict) -> sp.csr_matrix:
    """Sparse adjacency matrix for the path-star graph (both directions)."""
    n = len(node_to_idx)
    rows, cols = [], []
    for u, v in graph["edges_forward"] + graph["edges_reverse"]:
        rows.append(node_to_idx[u])
        cols.append(node_to_idx[v])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


# ──────────────────────────────────────────────────────────────────────────────
# Extended geometric diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def compute_eigenspace_gap(model, graph: dict, vocab: dict, k: int = 64) -> dict:
    """Projects node embeddings onto top-k adjacency eigenvectors."""
    node_tokens, node_to_idx, vocab_ids = _build_graph_node_index(graph, vocab)
    A = _build_adjacency(graph, node_to_idx)
    n_nodes = A.shape[0]
    k = min(k, n_nodes - 2)

    eigenvalues, V_k = spla.eigsh(A.astype(np.float64), k=k, which="LM")
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    V_k = V_k[:, order]

    E = model.emb.weight.detach().cpu().float().numpy()[vocab_ids]
    E_c = E - E.mean(axis=0)
    total_energy = np.linalg.norm(E_c, "fro") ** 2 + 1e-12

    per_eigvec_energy = np.zeros(k)
    for i in range(k):
        v = V_k[:, i : i + 1]
        proj = v @ (v.T @ E_c)
        per_eigvec_energy[i] = np.linalg.norm(proj, "fro") ** 2 / total_energy

    P = V_k @ (V_k.T @ E_c)
    gap = np.linalg.norm(E_c - P, "fro") ** 2 / total_energy

    return {
        "gap": float(gap),
        "captured_energy": float(1.0 - gap),
        "k": int(k),
        "n_nodes": int(n_nodes),
        "eigenvalues": eigenvalues.tolist(),
        "per_eigvec_energy": per_eigvec_energy.tolist(),
    }


def compute_stable_rank(model, graph: dict, vocab: dict) -> dict:
    """stable_rank = ||E||_F² / ||E||_2² — effective dimensionality."""
    _, _, vocab_ids = _build_graph_node_index(graph, vocab)
    E = model.emb.weight.detach().cpu().float().numpy()[vocab_ids]
    E_c = E - E.mean(axis=0)
    sv = np.linalg.svd(E_c, compute_uv=False)
    frob_sq = float((sv ** 2).sum())
    spec_sq = float(sv[0] ** 2)
    return {"stable_rank": frob_sq / (spec_sq + 1e-30), "singular_values": sv.tolist()}


def compute_gram_spectrum(model, graph: dict, vocab: dict) -> dict:
    """Gram matrix eigenspectrum: spectral gap ratio and top-5 energy fraction."""
    _, _, vocab_ids = _build_graph_node_index(graph, vocab)
    E = model.emb.weight.detach().cpu().float().numpy()[vocab_ids]
    E_c = E - E.mean(axis=0)
    ev = np.maximum(np.linalg.eigvalsh(E_c.T @ E_c)[::-1].copy(), 0)
    ev_sum = ev.sum() + 1e-30
    spectral_gap_ratio = float(ev[0] / (ev[1] + 1e-30)) if len(ev) > 1 else float("inf")
    top5_energy_fraction = float(ev[:5].sum() / ev_sum)
    return {
        "spectral_gap_ratio": spectral_gap_ratio,
        "top5_energy_fraction": top5_energy_fraction,
        "eigenvalues": ev.tolist(),
    }


def compute_principal_angles(model, graph: dict, vocab: dict, k: int = 64) -> dict:
    """Principal angles between embedding subspace and adjacency Laplacian eigenspace."""
    node_tokens, node_to_idx, vocab_ids = _build_graph_node_index(graph, vocab)
    A = _build_adjacency(graph, node_to_idx)
    n_nodes = A.shape[0]
    k = min(k, n_nodes - 2)

    _, V_k = spla.eigsh(A.astype(np.float64), k=k, which="LM")

    E = model.emb.weight.detach().cpu().float().numpy()[vocab_ids]
    E_c = E - E.mean(axis=0)
    Q, _ = np.linalg.qr(E_c, mode="reduced")
    cosines = np.clip(np.linalg.svd(V_k.T @ Q, compute_uv=False), 0.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    return {
        "min_angle_deg": float(angles.min()),
        "mean_angle_deg": float(angles.mean()),
        "angles_deg": angles.tolist(),
    }


def _geometric_battery(X: np.ndarray, V_k: np.ndarray) -> dict:
    """Full geometric battery on activation matrix X [n, d] vs eigenvectors V_k [n, k]."""
    X_c = X - X.mean(axis=0)
    total = np.linalg.norm(X_c, "fro") ** 2 + 1e-12

    P = V_k @ (V_k.T @ X_c)
    gap = float(np.linalg.norm(X_c - P, "fro") ** 2 / total)

    sv = np.linalg.svd(X_c, compute_uv=False)
    stable_rank = float((sv ** 2).sum() / (sv[0] ** 2 + 1e-30))

    ev = np.maximum(np.linalg.eigvalsh(X_c.T @ X_c)[::-1].copy(), 0)
    ev_sum = ev.sum() + 1e-30
    gram_gap_ratio = float(ev[0] / (ev[1] + 1e-30)) if len(ev) > 1 else float("inf")
    gram_top5 = float(ev[:5].sum() / ev_sum)

    Q, _ = np.linalg.qr(X_c, mode="reduced")
    cosines = np.clip(np.linalg.svd(V_k.T @ Q, compute_uv=False), 0.0, 1.0)
    angles = np.degrees(np.arccos(cosines))

    return {
        "eigenspace_gap": gap,
        "stable_rank": stable_rank,
        "gram_gap_ratio": gram_gap_ratio,
        "gram_top5_fraction": gram_top5,
        "min_principal_angle": float(angles.min()),
        "mean_principal_angle": float(angles.mean()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Residual stream extraction (for layer-wise probing)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_residual_streams(
    model, vocab: dict, leaves: list, device, max_seq_len: int
) -> tuple[dict, list]:
    """
    For each leaf, run [PATH] <leaf> [SEP] <pad>... and capture the residual stream
    at the leaf-token position (index 1) after each layer boundary.

    Returns (streams dict {layer_name: ndarray [n_leaves, d_model]}, layer_names).
    """
    model.eval()
    n_layers = len(model.blocks)
    leaf_pos = 1
    layer_names = ["emb"] + [f"block_{i}" for i in range(n_layers)] + ["ln_f"]
    storage: dict[str, list] = {name: [] for name in layer_names}

    for leaf in leaves:
        seq = (
            [vocab["path_id"], vocab["token2id"][leaf], vocab["sep_id"]]
            + [vocab["pad_id"]] * (max_seq_len - 3)
        )
        ids = torch.tensor(seq[:max_seq_len], dtype=torch.long, device=device).unsqueeze(0)
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)

        x = model.emb(ids) + model.pos_emb(pos)
        storage["emb"].append(x[0, leaf_pos].cpu().float().numpy())

        for i, block in enumerate(model.blocks):
            x = block(x)
            storage[f"block_{i}"].append(x[0, leaf_pos].cpu().float().numpy())

        storage["ln_f"].append(model.ln_f(x)[0, leaf_pos].cpu().float().numpy())

    return {k: np.array(v) for k, v in storage.items()}, layer_names


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1.5 — Translation consistency metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_translation_consistency(model, vocab: dict, graph: dict) -> dict:
    """
    Phase 1.5 Experiment A: Compute hop-delta vectors and measure global translation consistency.

    Returns:
      global_consistency[k]   – mean cosine sim to per-hop mean direction
      mean_global_consistency  – scalar mean over all hop levels
      inter_arm_angle_std      – std of pairwise angles between per-arm mean directions (degrees)
      delta_vecs               – [n_arms, path_length, d_model] normalized hop deltas
    """
    # Use raw emb.weight (not normalized) to match Stage 1 architecture
    E_raw = model.emb.weight.detach().cpu().float().numpy()
    # Normalize per-row for delta direction analysis (geometry is direction-based)
    norms = np.linalg.norm(E_raw, axis=1, keepdims=True)
    E_norm = E_raw / np.where(norms > 1e-12, norms, 1.0)
    path_length = len(graph["paths"][0]) - 1
    n_arms = len(graph["paths"])

    # Compute normalized hop-delta vectors
    delta_vecs = np.zeros((n_arms, path_length, E_norm.shape[1]))
    for arm_idx, path in enumerate(graph["paths"]):
        for k in range(path_length):
            v_k = E_norm[vocab["token2id"][path[k]]]
            v_kp1 = E_norm[vocab["token2id"][path[k + 1]]]
            d = v_kp1 - v_k
            norm = np.linalg.norm(d)
            delta_vecs[arm_idx, k] = d / (norm + 1e-12)

    # Level 1: global consistency per hop level
    global_consistency = np.zeros(path_length)
    mu_per_hop = np.zeros((path_length, E_norm.shape[1]))
    for k in range(path_length):
        dvk = delta_vecs[:, k, :]          # [n_arms, d_model]
        mu_k = dvk.mean(axis=0)
        mu_k = mu_k / (np.linalg.norm(mu_k) + 1e-12)
        mu_per_hop[k] = mu_k
        norms = np.linalg.norm(dvk, axis=1, keepdims=True) + 1e-12
        global_consistency[k] = float((dvk / norms @ mu_k).mean())

    mean_global_consistency = float(global_consistency.mean())

    # Level 2: inter-arm angle std (pairwise angles between per-arm mean directions)
    arm_mean_dirs = np.zeros((n_arms, E_norm.shape[1]))
    for arm_idx in range(n_arms):
        mu = delta_vecs[arm_idx].mean(axis=0)
        arm_mean_dirs[arm_idx] = mu / (np.linalg.norm(mu) + 1e-12)

    # Pairwise cosine similarities → angles
    cos_mat = arm_mean_dirs @ arm_mean_dirs.T
    cos_mat = np.clip(cos_mat, -1.0, 1.0)
    angles_mat = np.degrees(np.arccos(cos_mat))
    # Upper triangle only (exclude diagonal)
    triu_idx = np.triu_indices(n_arms, k=1)
    pairwise_angles = angles_mat[triu_idx]
    inter_arm_angle_std = float(pairwise_angles.std()) if len(pairwise_angles) > 0 else 0.0

    return {
        "global_consistency": global_consistency.tolist(),
        "mean_global_consistency": mean_global_consistency,
        "inter_arm_angle_std": inter_arm_angle_std,
        "delta_vecs_shape": list(delta_vecs.shape),
    }


def compute_svd_truncation_curve(
    model, vocab: dict, graph: dict, test_leaves: set, device,
    k_values: list[int] | None = None,
) -> dict:
    """
    Phase 1.5 Experiment B: Evaluate hardest_token_acc vs SVD rank k (no retraining).

    Uses the embedding matrix for the SVD but reconstructs ONLY the node embeddings
    while replacing ALL embeddings (both input and weight-tied output) with the truncated
    version, so the two uses stay consistent.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]

    _, _, vocab_ids = _build_graph_node_index(graph, vocab)
    E_full = model.emb.weight.detach().cpu().float().numpy()  # [vocab_size, d_model]

    # SVD of the full embedding matrix (centered)
    E_c = E_full - E_full.mean(axis=0)
    U, S, Vt = np.linalg.svd(E_c, full_matrices=False)

    original_weight = model.emb.weight.data.clone()
    full_rank_acc = eval_hardest_token_acc(model, vocab, graph, test_leaves, device)
    results = {}

    for k in k_values:
        k_actual = min(k, len(S))
        # Rank-k reconstruction of the full embedding matrix
        E_k = U[:, :k_actual] @ np.diag(S[:k_actual]) @ Vt[:k_actual, :] + E_full.mean(axis=0)

        model.emb.weight.data.copy_(
            torch.tensor(E_k, dtype=torch.float32, device=original_weight.device)
        )
        acc = eval_hardest_token_acc(model, vocab, graph, test_leaves, device)
        results[k_actual] = float(acc)

    # Restore original
    model.emb.weight.data.copy_(original_weight)

    # Find k* = smallest k where accuracy drops < 5pp from full-rank
    k_star = None
    for k in sorted(results.keys()):
        if results[k] >= full_rank_acc - 0.05:
            k_star = k
            break

    return {
        "k_values": sorted(results.keys()),
        "accuracies": [results[k] for k in sorted(results.keys())],
        "full_rank_acc": float(full_rank_acc),
        "k_star": k_star,
        "singular_values": S.tolist(),
    }


def compute_linear_probes(model, vocab: dict, graph: dict, train_leaves: set, test_leaves: set,
                          seed: int = 42) -> dict:
    """
    Phase 1.5 Experiment C: Linear probes for arm ID and hop depth from frozen E_norm.

    Uses a stratified 75/25 random split of ALL nodes (not the train/test leaf split),
    ensuring all arm labels appear in both train and test sets. This tests whether arm
    identity and hop depth are linearly decodable from the embedding geometry.
    """
    from sklearn.linear_model import LogisticRegression

    E_raw = model.emb.weight.detach().cpu().float().numpy()
    norms = np.linalg.norm(E_raw, axis=1, keepdims=True)
    E_norm_np = E_raw / np.where(norms > 1e-12, norms, 1.0)

    # Build feature matrix with labels for ALL nodes
    X_all, arm_labels, depth_labels = [], [], []
    for arm_idx, path in enumerate(graph["paths"]):
        for depth_idx, tok in enumerate(path[1:]):  # skip root
            X_all.append(E_norm_np[vocab["token2id"][tok]])
            arm_labels.append(arm_idx)
            depth_labels.append(depth_idx)

    X_all = np.array(X_all)
    arm_labels = np.array(arm_labels)
    depth_labels = np.array(depth_labels)

    # Stratified split by arm (each arm contributes to both train and test)
    rng = np.random.default_rng(seed)
    path_length = len(graph["paths"][0]) - 1
    n_arms = len(graph["paths"])
    n_test_per_arm = max(1, path_length // 4)  # hold out ~25% of depth levels per arm

    is_test = np.zeros(len(X_all), dtype=bool)
    for arm_idx in range(n_arms):
        arm_mask = arm_labels == arm_idx
        arm_indices = np.where(arm_mask)[0]
        test_indices = rng.choice(arm_indices, size=n_test_per_arm, replace=False)
        is_test[test_indices] = True

    X_train, X_test = X_all[~is_test], X_all[is_test]
    arm_train, arm_test = arm_labels[~is_test], arm_labels[is_test]
    depth_train, depth_test = depth_labels[~is_test], depth_labels[is_test]

    probe_arm = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_train, arm_train)
    probe_depth = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_train, depth_train)

    arm_probe_acc = float(probe_arm.score(X_test, arm_test))
    depth_probe_acc = float(probe_depth.score(X_test, depth_test))

    return {
        "arm_probe_acc": arm_probe_acc,
        "depth_probe_acc": depth_probe_acc,
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
        "n_arms": n_arms,
        "n_depth_classes": path_length,
    }
