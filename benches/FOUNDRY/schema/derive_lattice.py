#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Lattice Mould · 义素格模
    FOUNDRY intake fixture I：十六基格不是抄来的，是从引擎字节里浇出来的。
    The 16-base schema is DERIVED from cognitive_engine.py at run time —
    hand-copied tables go stale by construction (FOUNDLING staleness clause).

用法:
    python3 benches/FOUNDRY/schema/derive_lattice.py           # 浇模 → lattice16.json
    python3 benches/FOUNDRY/schema/derive_lattice.py --check   # 漂移审计：重浇并比对

退出码（--check）: 0 = 模与引擎同源 · 1 = 漂移（格模过期，拒绝引用）
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from cognitive_engine import SemeBasedCognitiveEngine  # noqa: E402

SCHEMA_PATH = os.path.join(HERE, "lattice16.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def cast():
    """重浇一格：从引擎当前字节读出基序、域、词库向量。"""
    eng = SemeBasedCognitiveEngine()
    bases = list(eng.basis_dict.keys())
    dims = len(bases)
    schema = {
        "engine": {"path": "cognitive_engine.py", "sha256": sha256(os.path.join(ROOT, "cognitive_engine.py"))},
        "dims": dims,
        "bases": [
            {
                "index": i,
                "name": b,
                # 前 10 基：市井域 · 后 6 基：叙事域（雷荷波扩展）
                "domain": "market" if i < 10 else "narrative",
            }
            for i, b in enumerate(bases)
        ],
        "invariants": {
            "coefficient_range": [-1.0, 1.0],
            "law_I": "每个词皆可分解；抵抗者候其基（零向量 = REFUSED，不是 0 分）",
            "law_IV": "L(E|S) 有 0.01 之底，注释本身没有",
        },
        "vocab": {w: dict(v) for w, v in eng.vocab.items()},
    }
    return schema


def main():
    check = "--check" in sys.argv[1:]
    fresh = cast()
    if not check:
        with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
            json.dump(fresh, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"✔ 模已浇成 · lattice16.json · dims={fresh['dims']} "
              f"· engine {fresh['engine']['sha256'][:16]}… · vocab={len(fresh['vocab'])}")
        return 0
    if not os.path.exists(SCHEMA_PATH):
        print("✘ REFUSED · 无模可审 — lattice16.json 不存在，先浇模")
        return 1
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        held = json.load(f)
    if held == fresh:
        print(f"✔ 模与引擎同源 · engine {fresh['engine']['sha256'][:16]}… · 无漂移")
        return 0
    # 精确定位漂移点，审计才可信
    print(f"✘ DRIFT · lattice16.json 与引擎字节不再同源")
    if held.get("engine", {}).get("sha256") != fresh["engine"]["sha256"]:
        print(f"    engine sha: held {held.get('engine',{}).get('sha256','?')[:16]}… vs fresh {fresh['engine']['sha256'][:16]}…")
    hb = [b["name"] for b in held.get("bases", [])]
    fb = [b["name"] for b in fresh["bases"]]
    if hb != fb:
        print(f"    bases: added {set(fb)-set(hb) or '—'} · removed {set(hb)-set(fb) or '—'} · renumbered {hb!=fb and (set(hb)&set(fb)) or '—'}")
    hv, fv = held.get("vocab", {}), fresh["vocab"]
    for w in sorted(set(hv) | set(fv)):
        if hv.get(w) != fv.get(w):
            print(f"    vocab[{w}]: {'MISSING-fresh' if w not in fv else 'MISSING-held' if w not in hv else 'coeff-changed'}")
    print("    → 重浇：python3 benches/FOUNDRY/schema/derive_lattice.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
