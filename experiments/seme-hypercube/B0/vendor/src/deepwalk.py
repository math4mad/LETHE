"""
DeepWalk implementation for C3 (pre-trained frozen embeddings).

Uses random walks on the path-star graph + gensim Word2Vec.
"""
from __future__ import annotations

import random
import numpy as np


def generate_random_walks(
    graph: dict,
    walk_length: int = 10,
    walks_per_node: int = 40,
    seed: int = 42,
) -> list[list[str]]:
    """Generate random walks on the path-star graph (undirected)."""
    rng = random.Random(seed)

    # Build adjacency list (undirected)
    adj: dict[str, list[str]] = {}
    for u, v in graph["edges_forward"] + graph["edges_reverse"]:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # All nodes
    all_nodes = list(adj.keys())

    walks = []
    for _ in range(walks_per_node):
        rng.shuffle(all_nodes)
        for start in all_nodes:
            walk = [start]
            for _ in range(walk_length - 1):
                current = walk[-1]
                neighbors = adj.get(current, [])
                if not neighbors:
                    break
                walk.append(rng.choice(neighbors))
            walks.append(walk)

    return walks


def run_deepwalk(
    graph: dict,
    vocab: dict,
    d_model: int,
    walk_length: int = 10,
    walks_per_node: int = 40,
    window: int = 5,
    seed: int = 42,
    min_count: int = 1,
    workers: int = 4,
) -> np.ndarray:
    """
    Run DeepWalk on the path-star graph and return an embedding matrix
    of shape [vocab_size, d_model] aligned to the vocab.

    Non-node tokens (special tokens) are initialized to zero.
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError(
            "gensim is required for DeepWalk (C3). Install with: pip install gensim"
        )

    walks = generate_random_walks(graph, walk_length=walk_length, walks_per_node=walks_per_node, seed=seed)

    model = Word2Vec(
        sentences=walks,
        vector_size=d_model,
        window=window,
        min_count=min_count,
        sg=1,          # skip-gram (standard DeepWalk)
        workers=workers,
        seed=seed,
        epochs=1,
    )

    vocab_size = vocab["vocab_size"]
    emb_matrix = np.zeros((vocab_size, d_model), dtype=np.float32)

    for token, idx in vocab["token2id"].items():
        if token in model.wv:
            emb_matrix[idx] = model.wv[token]
        # Special tokens and any missing tokens stay at zero

    return emb_matrix


def validate_pretrained_embeddings(emb_matrix: np.ndarray, vocab: dict, graph: dict, threshold: float = 0.7) -> dict:
    """
    Validate C3 pre-trained embeddings by checking arm_cluster_acc > threshold.
    Returns a dict with acc and pass/fail.
    """
    n_arms = len(graph["paths"])

    node_ids, arm_labels = [], []
    for arm_idx, path in enumerate(graph["paths"]):
        for tok in path[1:]:
            node_ids.append(vocab["token2id"][tok])
            arm_labels.append(arm_idx)

    E = emb_matrix[node_ids]
    labels = np.array(arm_labels)

    norms = np.linalg.norm(E, axis=1, keepdims=True)
    # Handle zero-norm rows (special token embeddings that weren't in DeepWalk)
    E_norm = E / np.where(norms > 1e-12, norms, 1.0)

    centroids = np.stack([E_norm[labels == i].mean(axis=0) for i in range(n_arms)])
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(c_norms > 1e-12, c_norms, 1.0)

    sims = E_norm @ centroids.T
    preds = sims.argmax(axis=1)
    acc = float((preds == labels).mean())

    return {
        "arm_cluster_acc": acc,
        "threshold": threshold,
        "pass": acc >= threshold,
    }
