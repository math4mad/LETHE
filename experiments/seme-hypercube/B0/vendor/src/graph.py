"""Graph construction and vocabulary utilities for the path-star task."""


def build_path_star_graph(degree: int, path_length: int) -> dict:
    root = "v_root"
    edges_forward = []
    paths = []

    for arm in range(degree):
        path_nodes = [root]
        for hop in range(1, path_length + 1):
            node = f"v_{arm}_{hop}"
            path_nodes.append(node)
            edges_forward.append((path_nodes[-2], path_nodes[-1]))
        paths.append(path_nodes)

    edges_reverse = [(b, a) for (a, b) in edges_forward]
    return {
        "root": root,
        "paths": paths,
        "edges_forward": edges_forward,
        "edges_reverse": edges_reverse,
    }


def build_vocab(graph_degree: int, path_length: int) -> dict:
    """Build deterministic vocab from graph shape."""
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[EDGE]", "[PATH]"]
    node_tokens = ["v_root"]
    for arm in range(graph_degree):
        for hop in range(1, path_length + 1):
            node_tokens.append(f"v_{arm}_{hop}")

    all_tokens = special_tokens + node_tokens
    token2id = {t: i for i, t in enumerate(all_tokens)}
    id2token = {i: t for t, i in token2id.items()}

    return {
        "token2id": token2id,
        "id2token": id2token,
        "vocab_size": len(all_tokens),
        "pad_id": token2id["[PAD]"],
        "bos_id": token2id["[BOS]"],
        "eos_id": token2id["[EOS]"],
        "sep_id": token2id["[SEP]"],
        "edge_id": token2id["[EDGE]"],
        "path_id": token2id["[PATH]"],
        "root_id": token2id["v_root"],
    }


def encode_edge_example(u: str, v: str, vocab: dict) -> list[int]:
    """[EDGE] u [SEP] v [EOS]"""
    return [
        vocab["edge_id"],
        vocab["token2id"][u],
        vocab["sep_id"],
        vocab["token2id"][v],
        vocab["eos_id"],
    ]


def encode_path_example(path: list[str], vocab: dict) -> list[int]:
    """
    [PATH] leaf [SEP] hop_{L-1} ... root [EOS]
    path = [root, hop_1, ..., leaf] — reversed to leaf-to-root in target.
    """
    leaf = path[-1]
    reverse_interior = list(reversed(path[:-1]))
    return (
        [vocab["path_id"], vocab["token2id"][leaf], vocab["sep_id"]]
        + [vocab["token2id"][n] for n in reverse_interior]
        + [vocab["eos_id"]]
    )


def build_train_test_leaves(graph: dict, train_leaf_fraction: float, seed: int = 42) -> tuple[set, set]:
    """
    Split arm leaves into train/test sets.
    Deterministic slice (first n_train leaves are train) to match original Stage 1 notebook.
    seed parameter kept for API compatibility but not used.
    """
    all_leaves = [path[-1] for path in graph["paths"]]
    n_train = int(len(all_leaves) * train_leaf_fraction)
    return set(all_leaves[:n_train]), set(all_leaves[n_train:])
