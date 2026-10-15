#!/usr/bin/env python3
"""FSSSS 度量聚合器 —— 三样交付律之「verify 命令」的输入生成器。

正本: report_*.json（各实验脚本的实测产物；重跑需 venv132 +
      external/ConceptualSpaces-1.3.2-py3，见 README「运行」）。
派生: metrics.json（本脚本输出，纯标准库，check_gates.py 的输入）。

用法:
    python3 make_metrics.py          # 读 report_*.json，写 metrics.json
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent


def ratio(text):
    """'7/8' -> 0.875；'10/10' -> 1.0。"""
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", str(text))
    if not m:
        raise ValueError(f"无法解析比例: {text!r}")
    return int(m.group(1)) / int(m.group(2))


def main():
    bridge = json.loads((HERE / "report_c_bridge.json").read_text())
    pcs = json.loads((HERE / "report_pcs_prototype.json").read_text())

    # ② 单步后验 argmax 一致（PCS vs 园引擎）
    ss = pcs["single_step"]
    ok = sum(1 for v in ss.values() if v.get("argmax") == v.get("garden_argmax"))
    single_agree = ok / len(ss)

    # 序贯轨迹：按词对齐，比较终态后验 argmax；并算跨空间极差（退化检测）
    sp = {e["word"]: e for e in pcs["sequential_pcs"]}
    sg = {e["word"]: e for e in pcs["sequential_garden"]}
    common = [w for w in sp if w in sg]
    seq_ok, spreads = 0, []
    for w in common:
        pa, pb = sp[w]["posterior"], sg[w]["posterior"]
        if max(pa, key=pa.get) == max(pb, key=pb.get):
            seq_ok += 1
        vals = list(pa.values())
        spreads.append(max(vals) - min(vals))
    seq_agree = seq_ok / len(common)
    mean_spread = sum(spreads) / len(spreads)

    # 模型选择判别力：留出（test）边际似然最优 c 与最差 c 的证据差（nats）。
    # 只用 test —— 选 c 与评 c 必须分开，用 total 会混入 train 对大 c 的单调偏好。
    tests = [v["test"] for v in pcs["c_scan"].values()]
    margin = max(tests) - min(tests)
    # 辅助信息（不进门限）：总证据口径的判别力与最优 c
    totals = [v["total"] for v in pcs["c_scan"].values()]
    total_margin = max(totals) - min(totals)

    metrics = {
        "bench": "FSSSS",
        "source_reports": ["report_c_bridge.json", "report_pcs_prototype.json"],
        "bridge_rank_agreement": ratio(bridge["rank_agreement"]),
        "bridge_anchor_selfconsistency": ratio(bridge["anchor_selfconsistency"]),
        "pcs_norm_rel_err": pcs["normalization_2d"]["rel_err"],
        "pcs_single_step_agreement": single_agree,
        "pcs_sequential_agreement": seq_agree,
        "pcs_posterior_spread_mean": mean_spread,
        "pcs_model_selection_margin_nats": margin,
        "pcs_model_selection_total_margin_nats": total_margin,
        "pcs_best_c": pcs["best_c"],
        "pcs_best_c_criterion": "argmax of c_scan[*].test (held-out)",
        "pcs_log_evidence_at_c8": pcs["pcs_log_evidence"],
    }
    (HERE / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
