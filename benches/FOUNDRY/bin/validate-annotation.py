#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Intake Gate · 铸入门禁
    FOUNDRY intake fixture II：任何新铸义素注释，过此门才算入册。
    A candidate annotation is REFUSED (a first-class lane, Foundling §2.3)
    unless it survives four stations:

      1. 形制 · schema    — known bases, numeric, within [-1.0, 1.0]
      2. 律 I · decomposable — non-zero vector; what resists awaits its basis
      3. 复现 · reproduction — if the word is already in the ledger, the
         annotation must reproduce the engine's current vector (≤1e-6),
         else the mould has drifted and the draft is refused
      4. 试浇 · trial pour — new words are poured against the frozen ledger
         and the posterior tilt is reported (evidence, not verdict)

用法:
    python3 benches/FOUNDRY/bin/validate-annotation.py drafts/xxx.json [...]

退出码: 0 = 全部 ACCEPTED · 1 = 有 REFUSED（阴性结果是公民，不是错误）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from cognitive_engine import SemeBasedCognitiveEngine  # noqa: E402

EPS = 1e-6
SCHEMA = os.path.join(ROOT, "benches", "FOUNDRY", "schema", "lattice16.json")


def load_mould():
    if not os.path.exists(SCHEMA):
        sys.exit("✘ 无模 · lattice16.json 不存在 — 先跑 derive_lattice.py（律：模先于铸）")
    with open(SCHEMA, encoding="utf-8") as f:
        return json.load(f)


def encode(eng, vec):
    bases = list(eng.basis_dict.keys())
    x = np.zeros(len(bases))
    for b, c in vec.items():
        if b in bases:
            x[bases.index(b)] = c
    return x


def station_schema(draft, mould):
    errs = []
    known = {b["name"] for b in mould["bases"]}
    if "word" not in draft or "vector" not in draft:
        errs.append("缺 word 或 vector 字段")
        return errs
    if not isinstance(draft["vector"], dict) or not draft["vector"]:
        errs.append("vector 必须是非空对象 {基: 系数}")
        return errs
    for b, c in draft["vector"].items():
        if b not in known:
            errs.append(f"未知义素基 '{b}'（抵抗者候其基 — 先提案新基，勿私铸）")
        if not isinstance(c, (int, float)) or isinstance(c, bool):
            errs.append(f"'{b}' 系数非数: {c!r}")
        elif not (-1.0 <= float(c) <= 1.0):
            errs.append(f"'{b}' 系数 {c} 越出 [-1, 1]")
    return errs


def station_decompose(draft):
    if all(abs(float(c)) < EPS for c in draft.get("vector", {}).values()):
        return ["零向量：此词抵抗分解 — 律 I 之虚位，REFUSED 而非 0 分"]
    return []


def station_reproduce(eng, draft):
    w = draft["word"]
    if w not in eng.vocab:
        return []  # 新词，无账可复
    cur, cand = encode(eng, eng.vocab[w]), encode(eng, draft["vector"])
    diff = float(np.max(np.abs(cur - cand)))
    if diff > EPS:
        moved = {b: (round(float(cur[i]), 4), round(float(cand[i]), 4))
                 for i, b in enumerate(eng.basis_dict) if abs(cur[i] - cand[i]) > EPS}
        return [f"复现失败：与在账向量最大偏差 {diff:g} — {moved}（模与账已分，先重浇 lattice16.json）"]
    return []


def station_trial(eng, draft):
    """试浇：克隆引擎、注入新词、观察一步贝叶斯，报告概率倾侧。"""
    import copy
    if draft["word"] in eng.vocab:
        return None
    probe = copy.deepcopy(eng)
    probe.vocab[draft["word"]] = {b: float(c) for b, c in draft["vector"].items()}
    buf, real = [], sys.stdout
    sys.stdout = type("_Cap", (), {"write": lambda s, t: buf.append(t), "flush": lambda s: None})()
    try:
        probe.observe(draft["word"])
    finally:
        sys.stdout = real
    return {name: round(sp["prior_prob"], 4) for name, sp in probe.concept_spaces.items()}


def vet(path, mould):
    eng = SemeBasedCognitiveEngine()
    with open(path, encoding="utf-8") as f:
        draft = json.load(f)
    reasons = station_schema(draft, mould)
    reasons += station_decompose(draft) if not station_schema(draft, mould) else []
    verdict = "REFUSED" if reasons else "ACCEPTED-DRAFT"
    if not reasons:
        reasons += station_reproduce(eng, draft)
        if reasons:
            verdict = "REFUSED"
    tilt = None
    if not reasons:
        tilt = station_trial(eng, draft)
    return draft.get("word", "?"), verdict, reasons, tilt


def main():
    mould = load_mould()
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not paths:
        sys.exit(f"用法: {os.path.relpath(sys.argv[0], ROOT)} <draft.json> [...]")
    any_refused = False
    for p in paths:
        word, verdict, reasons, tilt = vet(p, mould)
        mark = "⊘" if verdict == "REFUSED" else "⚏"
        print(f"{mark} {verdict:15s} 【{word}】  ← {os.path.relpath(p, ROOT)}")
        for r in reasons:
            print(f"    · {r}")
        if tilt:
            top = sorted(tilt.items(), key=lambda kv: -kv[1])
            lead = " / ".join(f"{k} {v:.2%}" for k, v in top[:3])
            print(f"    试浇倾侧: {lead}")
        any_refused |= verdict == "REFUSED"
    print(f"\n台账之模: engine {mould['engine']['sha256'][:16]}… · dims {mould['dims']} · 词 {len(mould['vocab'])}")
    print("REFUSED 是车道，不是耻辱。" if any_refused else "全数过门。")
    return 1 if any_refused else 0


if __name__ == "__main__":
    sys.exit(main())
