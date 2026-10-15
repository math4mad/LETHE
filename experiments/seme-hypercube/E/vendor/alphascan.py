"""E 臂：α 插值扫描 —— emb(α) = α·DW + (1−α)·rnd_s。

预注册：benches/seme-hypercube/prereg.md §7（2026-10-15
冻结，先于首跑）。设计要点：

- DW：gensim skip-gram on Hamming 边集（to_orig_graph
  转换后），walk_length 10 / walks_per_node 40 / window 5
  / seed = 训练种子 —— 与 C3 每种子同源，故 α=1 ≡ C3
  （恒等端点，验证机制）。
- rnd_s：N(0,1)（np.random.default_rng(1000·i + seed)），
  全局行范数匹配到 DW 尺度（mean‖DW‖₂ / mean‖rnd‖₂）。
  gensim 原始尺度 ≪ N(0,1) 默认初始化，不匹配则 α=0
  条件注入的是「尺度」而非「结构」——尺度钉在 DW 轴上，
  α 纯控结构比例（防尺度-结构混淆）。
- 特殊 token 行不特判：α=1 时与 C3 同为零行（DeepWalk
  未覆盖特殊 token）；α<1 时自然填入。
"""
import numpy as np


def build_alpha_embedding(dw_emb: np.ndarray, alpha: float,
                           condition_i: int,
                           seed: int) -> np.ndarray:
    """emb(α)：形状同 dw_emb，(vocab_size, d_model)。

    condition_i = 档号 i（α = i/6），seed = 训练种子；
    随机源 rng(1000·i + seed) 使各档随机分量不同、
    同档跨种子不同（每种子独立实现）。
    """
    rng = np.random.default_rng(1000 * condition_i + seed)
    rnd = rng.standard_normal(dw_emb.shape)
    dw_scale = float(np.linalg.norm(dw_emb, axis=1).mean())
    rnd_scale = float(np.linalg.norm(rnd, axis=1).mean())
    rnd_s = rnd * (dw_scale / rnd_scale)
    return alpha * dw_emb + (1.0 - alpha) * rnd_s
