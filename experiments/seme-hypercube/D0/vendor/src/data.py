"""Dataset construction and batch sampling for the path-star task."""
import numpy as np
import torch

from .graph import encode_edge_example, encode_path_example


def build_dataset(
    graph: dict,
    vocab: dict,
    train_leaves: set,
    test_leaves: set,
    max_seq_len: int,
    pad_id: int,
) -> dict:
    """
    Build all examples. Edge examples cover ALL edges (both directions, all arms).
    Path examples are split: train leaves for training, test leaves held out.
    All sequences padded to max_seq_len.
    """

    def pad(seq):
        seq = list(seq)
        assert len(seq) <= max_seq_len, f"Sequence length {len(seq)} > max_seq_len {max_seq_len}"
        return seq + [pad_id] * (max_seq_len - len(seq))

    edge_seqs = [
        pad(encode_edge_example(u, v, vocab))
        for u, v in graph["edges_forward"] + graph["edges_reverse"]
    ]

    train_path_seqs, test_path_seqs = [], []
    for path in graph["paths"]:
        seq = pad(encode_path_example(path, vocab))
        (train_path_seqs if path[-1] in train_leaves else test_path_seqs).append(seq)

    return {
        "edge": torch.tensor(edge_seqs, dtype=torch.long),
        "train_path": torch.tensor(train_path_seqs, dtype=torch.long),
        "test_path": torch.tensor(test_path_seqs, dtype=torch.long),
    }


class MixedBatchSampler:
    """
    Samples mixed batches of edge and path examples.
    edge_path_mix_ratio=1.0 → 50% edge, 50% path per batch.
    """

    def __init__(self, dataset: dict, batch_size: int, edge_path_mix_ratio: float, seed: int = 42):
        self.edge = dataset["edge"]
        self.path = dataset["train_path"]
        n_edge = max(1, int(round(batch_size * edge_path_mix_ratio / (edge_path_mix_ratio + 1))))
        self.n_edge = n_edge
        self.n_path = max(1, batch_size - n_edge)
        self.rng = np.random.default_rng(seed)

    def sample_batch(self) -> torch.Tensor:
        ei = self.rng.integers(0, len(self.edge), size=self.n_edge)
        pi = self.rng.integers(0, len(self.path), size=self.n_path)
        return torch.cat([self.edge[ei], self.path[pi]], dim=0)
