#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Observation Ledger · 观察台账
    board.html 的家法：intent is hand-written (the OBSERVATIONS list below),
    but every lane is derived from bytes on the same beat — and a lane may say
    REFUSED and mean it (negative results are first-class citizens).

输出: website/observations.json — 由主页 ledger 区读取渲染。
重跑: python3 bin/observe-to-json.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from cognitive_engine import SemeBasedCognitiveEngine  # noqa: E402

# ── intent, hand-written · 意图，亲手所书 ─────────────────────────────
OBSERVATIONS = [
    {"evidence": "塑胶凳",                        "claim": "市井域基线：户外塑料拉向路边摊"},
    {"evidence": "煤气罐",                        "claim": "市井域强证据：户外现做压垮超市"},
    {"evidence": "我不要再被人摆布",               "claim": "叙事域首击：负义素（服从 −1.0）入册"},
    {"evidence": "这些残暴的欢愉，终将以残暴结局",  "claim": "叙事域次击：觉醒循环应越过 40%"},
    {"evidence": "菠萝",                          "claim": "阴性结果：词库外的词，门禁拒收（律 I 之虚位）"},
]
ARTIFACT_NOTES = [
    {"event": "self-match artifact", "status": "dead 2026-09 · kept as load-bearing wall",
     "detail": "每一词曾以 1.0000 锚定自身于所有球——锚点检索现已排除证据本身。"},
    {"event": "likelihood floor", "status": "standing law IV",
     "detail": "L(E|S) 以 0.01 为底而不归零：再死的球，一句台词亦可唤醒。"},
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    engine_path = os.path.join(ROOT, "cognitive_engine.py")
    eng = SemeBasedCognitiveEngine()
    steps = []
    for i, ob in enumerate(OBSERVATIONS, 1):
        w = ob["evidence"]
        rec = {"n": i, "evidence": w, "claim": ob["claim"]}
        if w not in eng.vocab:
            rec["lane"] = "REFUSED"
            rec["reason"] = "not in the semic lexicon — Law I: awaits its basis"
            steps.append(rec)
            continue
        before = {k: v["prior_prob"] for k, v in eng.concept_spaces.items()}
        maxsim = eng.calculate_maxsim_distance(w)
        iv = eng.encode_word(w)
        lk, posts, tot = {}, {}, 0.0
        for space, con in eng.concept_spaces.items():
            cv = np.zeros(len(eng.basis_dict))
            bn = list(eng.basis_dict.keys())
            for b, c in con["coefficients"].items():
                if b in bn:
                    cv[bn.index(b)] = c * con["attention_weights"].get(b, 1.0)
            sim = float(np.dot(cv, iv) / (np.linalg.norm(cv) * np.linalg.norm(iv) + 1e-8))
            lk[space] = max(sim, 0.01)
            posts[space] = lk[space] * con["prior_prob"]
            tot += posts[space]
        eng.observe(w)  # advances the priors, prints its own analysis
        after = {k: v["prior_prob"] for k, v in eng.concept_spaces.items()}
        rec.update({
            "lane": "OBSERVED",
            "maxsim": {s: {"score": round(v[0], 4), "anchor": v[1]} for s, v in maxsim.items()},
            "likelihood": {s: round(v, 4) for s, v in lk.items()},
            "prior": {s: round(before[s], 4) for s in before},
            "posterior": {s: round(after[s], 4) for s in after},
            "delta_pt": {s: round((after[s] - before[s]) * 100, 1) for s in after},
        })
        steps.append(rec)

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": {"path": "cognitive_engine.py", "sha256": sha256(engine_path)[:16] + "…"},
        "spaces": list(eng.concept_spaces.keys()),
        "steps": steps,
        "artifact_notes": ARTIFACT_NOTES,
        "final_priors": {k: round(v["prior_prob"], 4) for k, v in eng.concept_spaces.items()},
    }
    dest = os.path.join(ROOT, "website", "observations.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ledger painted →", dest)

if __name__ == "__main__":
    main()
