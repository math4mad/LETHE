from dataclasses import dataclass, field


@dataclass
class StageConfig:
    stage_name: str
    n_layer: int
    d_model: int
    n_head: int
    graph_degree: int
    path_length: int
    train_leaf_fraction: float
    train_steps: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    edge_path_mix_ratio: float  # edge:path
    notes: str = ""


STAGE_PRESETS: dict[str, StageConfig] = {
    "stage0_local_smoke": StageConfig(
        stage_name="stage0_local_smoke", n_layer=4, d_model=192, n_head=6,
        graph_degree=128, path_length=4, train_leaf_fraction=0.75,
        train_steps=10000, batch_size=16, grad_accum_steps=2,
        learning_rate=3e-4, weight_decay=0.01, edge_path_mix_ratio=1.0,
        notes="Fast local signal check",
    ),
    "stage1_local_core": StageConfig(
        stage_name="stage1_local_core", n_layer=8, d_model=256, n_head=8,
        graph_degree=512, path_length=6, train_leaf_fraction=0.75,
        train_steps=40000, batch_size=8, grad_accum_steps=4,
        learning_rate=2e-4, weight_decay=0.01, edge_path_mix_ratio=1.0,
        notes="Main low-cost fidelity run",
    ),
    "stage2_assoc_control": StageConfig(
        stage_name="stage2_assoc_control", n_layer=8, d_model=256, n_head=8,
        graph_degree=512, path_length=6, train_leaf_fraction=0.75,
        train_steps=40000, batch_size=8, grad_accum_steps=4,
        learning_rate=2e-4, weight_decay=0.01, edge_path_mix_ratio=1.0,
        notes="Embedding-frozen associative control",
    ),
    "stage3_cloud_confirm": StageConfig(
        stage_name="stage3_cloud_confirm", n_layer=12, d_model=384, n_head=8,
        graph_degree=1024, path_length=6, train_leaf_fraction=0.75,
        train_steps=50000, batch_size=8, grad_accum_steps=8,
        learning_rate=1.5e-4, weight_decay=0.01, edge_path_mix_ratio=1.0,
        notes="Optional one-shot larger confirmation",
    ),
}

# Success gates per stage
SUCCESS_GATES: dict[str, dict] = {
    "stage0_local_smoke": {
        "hardest_token_acc_min": 0.20,
        "final_auc_min": 0.60,
        "alignment_delta_min": 0.02,
    },
    "stage1_local_core": {
        "hardest_token_acc_min": 0.40,
        "final_auc_min": 0.80,
        "alignment_delta_min": 0.05,
    },
    "stage2_assoc_control": {
        "expect_drop_vs_stage1": True,
    },
    "stage3_cloud_confirm": {
        "hardest_token_acc_min": 0.60,
        "final_auc_min": 0.90,
        "alignment_delta_min": 0.10,
    },
}
