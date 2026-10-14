#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-TDV verify — 退出码即判据 (三样交付律之 verify)

用法: python3 verify_temporal_direction.py [metrics_temporal_direction.json]
退出码: 0 = 落账完整且三判据已裁决; 1 = 缺项/数值缺失/与 PREREG 旋钮不符
"""
import json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "metrics_temporal_direction.json")

required = {
    "experiment": "E-TDV temporal direction vector",
    "knobs": {"seed_fixed": 13, "shuffle_seeds": [14, 15, 16],
              "alpha_grid": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
              "lora_r": 16, "lora_alpha": 32, "epochs": 6, "lr": 0.0001,
              "adam_beta2": 0.95, "block": 128,
              "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
}
sections = ["vector", "K1_probe", "K2_timemachine", "K3_lora_projection"]

def main():
    try:
        m = json.load(open(path))
    except Exception as e:
        print(f"FAIL: 读账失败 {path}: {e}")
        return 1
    errs = []
    if m.get("experiment") != required["experiment"]:
        errs.append(f"experiment 不符: {m.get('experiment')!r}")
    for k, want in required["knobs"].items():
        got = m.get("knobs", {}).get(k)
        if got != want:
            errs.append(f"knob {k} 不符: {got!r} != {want!r}")
    for s in sections:
        if s not in m:
            errs.append(f"缺节: {s}")
    # 三判据字段完整性
    k1 = m.get("K1_probe", {})
    for f in ("rho_vtemporal", "rho_vshuf_ctrl", "pass", "verdict"):
        if f not in k1:
            errs.append(f"K1_probe 缺字段 {f}")
    k2 = m.get("K2_timemachine", {})
    if len(k2.get("mean_predicted_rank", [])) != 7:
        errs.append("K2 alpha 曲线长度 != 7")
    for f in ("monotone", "span", "pass", "verdict"):
        if f not in k2:
            errs.append(f"K2_timemachine 缺字段 {f}")
    k3 = m.get("K3_lora_projection", {})
    for f in ("proj_ord", "proj_shuf", "proj_shuf_mean", "pass", "verdict"):
        if f not in k3:
            errs.append(f"K3_lora_projection 缺字段 {f}")
    if len(k3.get("proj_shuf", [])) != 3:
        errs.append("K3 shuf 臂数 != 3")
    if errs:
        print("FAIL:")
        for e in errs:
            print("  -", e)
        return 1
    print(f"OK: E-TDV 落账完整 — K1 {'过' if k1['pass'] else '不过'} "
          f"(|ρ|={abs(k1['rho_vtemporal']):.3f} vs shuf {abs(k1['rho_vshuf_ctrl']):.3f}); "
          f"K2 {'过' if k2['pass'] else '不过'} (span={k2['span']:.2f}); "
          f"K3 {'过' if k3['pass'] else '不过'} (ord {k3['proj_ord']:+.4f} vs shuf {k3['proj_shuf_mean']:+.4f})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
