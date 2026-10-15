#!/usr/bin/env python3
"""bench 门限裁判 —— 三样交付律之「一条 verify 命令」。

正本: gates.json（门限）+ metrics.json（bench 产物）。逐项对表，任一未过即退出码 1；
verdict.json 为派生判词（人可读，附输入 sha，可追账进 LEDGER.md）。

用法:
    python3 check_gates.py [--bench DIR]                 # 读 DIR/{gates,metrics}.json，写 DIR/verdict.json
    python3 check_gates.py --gates G --metrics M         # 任意路径
    python3 check_gates.py --bench DIR --stage training  # 判 gates.json 之 training_gates.gates（第二批训练期门限）

两批门限：gates[ ] = 第一批装置不变量（默认）；
training_gates.gates[ ] = 第二批训练期门限（--stage training）。
training_gates.rules[ ] 为解释规则（含推翻线/无差异带），
仅记录供跑后对照，不参与裁判（互斥分支不能同时当硬门限）。

门限语法 (gates.json 之 gates[]):
    {"name": "primary_acc", "min": 0.40}                 # 值必须 >= 0.40
    {"name": "noise",       "max": 0.05}                 # 值必须 <= 0.05
    {"name": "ratio",       "min": 1, "max": 10}         # 区间
    {"name": "maxsim_by_space", "spread_min": 0.05}      # name 须是 metrics.json 里的
        # 字典键（如各空间 MaxSim）；极差 < 下限 = 恒报同值（非退化门限，1008 病）
    {"name": "assoc_control_acc", "expect_drop_vs": "primary_acc", "min_drop_bp": 50}
        # 对照条件须比参照低 >= 50bp（他们 stage2 expect_drop_vs_stage1 之推广）

判词标记: ✓ 过 / ✗ 不过 / ? 无法判（缺数据，按不过计 —— 保守）。
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def judge(rule, metrics):
    """返回 (ok, 判词)。ok=None 表示无法判（缺数据），按不过计。"""
    name = rule["name"]
    val = metrics.get(name)

    if "expect_drop_vs" in rule:
        ref = metrics.get(rule["expect_drop_vs"])
        a, b = num(val), num(ref)
        if a is None or b is None:
            return None, f"{name} 或参照 {rule['expect_drop_vs']} 缺失"
        drop_bp = (b - a) * 10000
        need = rule.get("min_drop_bp", 0)
        return drop_bp >= need, (
            f"{name}={a:.4g} 比 {rule['expect_drop_vs']}={b:.4g} 低 {drop_bp:.0f}bp（需 ≥{need}bp）"
        )

    if "spread_min" in rule or "spread_max" in rule:
        if not isinstance(val, dict) or not val:
            return None, f"{name} 缺失或不是非空字典"
        vals = [num(v) for v in val.values()]
        if any(v is None for v in vals):
            return None, f"{name} 内含非数值"
        spread = max(vals) - min(vals)
        if "spread_min" in rule and spread < rule["spread_min"]:
            return False, (
                f"{name} 极差 {spread:.4g} < 下限 {rule['spread_min']}"
                "（退化嫌疑：各空间恒报同值？1008 病）"
            )
        if "spread_max" in rule and spread > rule["spread_max"]:
            return False, f"{name} 极差 {spread:.4g} > 上限 {rule['spread_max']}"
        lo = rule.get("spread_min", "-∞")
        hi = rule.get("spread_max", "+∞")
        return True, f"{name} 极差 {spread:.4g} 在 [{lo}, {hi}]"

    a = num(val)
    if a is None:
        return None, f"{name} 缺失或非数值"
    lo, hi = rule.get("min"), rule.get("max")
    ok = (lo is None or a >= lo) and (hi is None or a <= hi)
    bounds = " ".join(s for s in [f">= {lo}" if lo is not None else "",
                                  f"<= {hi}" if hi is not None else ""] if s)
    return ok, f"{name} = {a:.4g}（需 {bounds}）"


def main():
    ap = argparse.ArgumentParser(description="bench 门限裁判（退出码即判据）")
    ap.add_argument("--bench", default=".", help="含 gates.json 与 metrics.json 的目录")
    ap.add_argument("--gates")
    ap.add_argument("--metrics")
    ap.add_argument("--stage", default="device",
                        choices=["device", "training"],
                        help="门限批：device（默认，第一批装置不变量）/ training（第二批训练期门限）")
    args = ap.parse_args()

    bench = Path(args.bench)
    gates_p = Path(args.gates) if args.gates else bench / "gates.json"
    metrics_p = Path(args.metrics) if args.metrics else bench / "metrics.json"
    verdict_p = bench / "verdict.json"

    for p in (gates_p, metrics_p):
        if not p.exists():
            sys.exit(f"check_gates: 缺文件 {p}")

    gates = json.loads(gates_p.read_text())
    metrics = json.loads(metrics_p.read_text())

    if args.stage == "training":
        rule_set = gates.get("training_gates", {}).get("gates", [])
        if not rule_set:
            sys.exit("check_gates: gates.json 无 training_gates.gates（第二批门限未冻结）")
        # 训练期产物形如 {"bench":..., "condition": "training", "metrics_flat": {...}}
        # ——门限键在 metrics_flat 包装下，展开到顶层再对表（1015 C2/C3 首跑实测）
        metrics = {**metrics, **metrics.get("metrics_flat", {})}
    else:
        rule_set = gates.get("gates", [])

    results, fails = [], 0
    for rule in rule_set:
        ok, note = judge(rule, metrics)
        why = rule.get("why", "")
        mark = {True: "✓", False: "✗", None: "?"}[ok]
        if ok is not True:
            fails += 1
        results.append({"name": rule["name"], "ok": ok, "note": note, "why": why})
        print(f"{mark} {note}" + (f"   [{why}]" if why else ""))

    verdict = "PASS" if fails == 0 else "FAIL"
    print("-" * 72)
    print(f"VERDICT: {verdict}  ({len(results) - fails}/{len(results)} 过)")

    payload = {
        "bench": metrics.get("bench", gates.get("bench")),
        "stage": args.stage,
        "verdict": verdict,
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gates_sha": sha256(gates_p),
        "metrics_sha": sha256(metrics_p),
        "results": results,
    }
    verdict_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"verdict.json 已写: {verdict_p}（gates@{payload['gates_sha']} × metrics@{payload['metrics_sha']}）")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
