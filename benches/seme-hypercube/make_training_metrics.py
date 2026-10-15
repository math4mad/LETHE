#!/usr/bin/env python3
"""训练期判读面聚合器（Kaggle 夜航 3-seed 汇总 → 判读面）。

正本：experiments/seme-hypercube/results/<臂>/<COND>_summary.json
      （Kaggle 回收的 3-seed 均值/min/max，逐 seed 明细在
      同目录 <COND>_seeds.json）。
派生：benches/seme-hypercube/metrics_training.json
      （check_gates.py --stage training 的判读输入）。

口径（锚原仓库 notebooks/02_fidelity_on_budget.ipynb
cell 38：加载 best checkpoint 后评估）：
- hardest_token_acc 门限 = best-checkpoint 值
  （best ckpt 按 hardest_token_acc 选取，其值即日志
  最大值；训练终态值另存 *_hardest_token_acc_final
  作辅助对照）。
- final_auc = arm_cluster_acc，本仓在训练终态模型上
  评估（原仓库在 best ckpt 上评估；ckpt 按回收纪律
  已清除，口径差在此诚实标注，不作事后追测）。
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = (HERE.parent.parent / "experiments"
           / "seme-hypercube" / "results")
ARMS = ["B0", "C1", "C2", "C3", "D0",
        "E0", "E1", "E2", "E3", "E4", "E5", "E6"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    flat = {"bench": "seme-hypercube", "stage": "training"}
    prov = {}
    for arm in ARMS:
        summary_p = RESULTS / arm / f"{arm}_summary.json"
        if not summary_p.exists():
            raise SystemExit(f"make_training_metrics: 缺正本 {summary_p}")
        summary = json.loads(summary_p.read_text())
        for k, v in summary.items():
            if v is None:
                continue
            flat[f"{arm}_{k}"] = v["mean"] if isinstance(v, dict) else v
        # 门限别名（原仓库 SUCCESS_GATES 形名，口径见 docstring）
        flat[f"{arm}_hardest_token_acc"] = flat[
            f"{arm}_hardest_token_acc_best"]
        flat[f"{arm}_final_auc"] = flat[f"{arm}_arm_cluster_acc"]
        prov[arm] = {
            "source": str(summary_p),
            "sha256": sha256(summary_p),
            "per_seed_detail": str(
                RESULTS / arm / f"{arm}_seeds.json"),
        }
    # 相对加速比（门限形式 2026-10-15 主人裁定变更）：
    # C1_steps_to_90 / C3_steps_to_90 ≥ 10×。
    # C1 未达 90% 时以训练步数上界（40000）作分子 → 下界。
    c1_steps = flat.get("C1_steps_to_90")
    c3_steps = flat.get("C3_steps_to_90")
    if c1_steps is None:
        c1_steps = 40000  # 训练步数上界（C1 未达 90% 线）
    if c3_steps:
        flat["C3_accel_ratio_min"] = c1_steps / c3_steps
    flat["_provenance"] = {
        "aggregator": "make_training_metrics.py",
        "arms": prov,
        "口径说明": (
            "hardest_token_acc = best-checkpoint 值（原仓库 "
            "cell 38 语义）；final_auc = arm_cluster_acc 于训练"
            "终态模型（原仓库为 best ckpt 上评估，ckpt 未回收，"
            "口径差诚实标注）"),
    }
    out = HERE / "metrics_training.json"
    out.write_text(json.dumps(flat, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    for arm in ARMS:
        print(f"  {arm}: hardest(best)={flat[f'{arm}_hardest_token_acc']:.4f} "
              f"final_auc={flat[f'{arm}_final_auc']:.4f} "
              f"hop1={flat[f'{arm}_hop1_alignment']:.4f} "
              f"hop2to6={flat[f'{arm}_hop2to6_alignment']:.4f}")


if __name__ == "__main__":
    main()
