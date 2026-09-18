#!/usr/bin/env python3
# C3 · 先验指针呼吸实验 — the "main horse" the prior-pointer article asked for.
# Drives the PARK'S OWN engine (cognitive_engine.SemeBasedCognitiveEngine, 16-seme lattice)
# with time-modulated priors α(t), β(t) and answers, in numbers:
#   Q1 does posterior entropy breathe (smooth→flat, sharp→peaked)?
#   Q2 does the argmax sphere flip at prior flips (the director changes scripts)?
#   Q3 does the MaxSim ANCHOR word inside each sphere change with the prior (not just
#      probability magnitude — the geometry itself responds)?
# Output: report + figure into benches/PRIOR_POINTER/. CPU-only, engine bytes untouched.
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cognitive_engine import SemeBasedCognitiveEngine

OUT = os.path.dirname(os.path.abspath(__file__))
A_AMP, OMEGA = 1.2, 1.0
EV = ["塑胶凳", "购物车", "煤气罐", "货架"]


FLOOR = 0.05   # concentration must stay positive; A>1 oscillators can otherwise dip negative
               # (found 2026-09-18 by C3's first run — the same latent bug lived in the marimo panel)
def alpha(t): return max(FLOOR, (1 + A_AMP * np.cos(OMEGA * t)) + 0.05)
def beta(t):  return max(FLOOR, (1 - A_AMP * np.cos(OMEGA * t)) + 0.05)


def posterior(engine, word, t):
    """Same law as the marimo panel: likelihood frozen, temperature modulated by the pointer."""
    a, b = alpha(t), beta(t)
    v = engine.encode_word(word)
    scores = {}
    for cname, sp in engine.concept_spaces.items():
        cv = engine.get_concept_vector(cname)
        c = float(np.dot(v, cv) / (np.linalg.norm(v) * np.linalg.norm(cv) + 1e-8))
        temp = 0.25 + 1.5 * a / (a + b)   # SMOOTH→hot→uniform, SHARP→cold→polarised (sign fixed 2026-09-18 after first run exposed the inversion)
        scores[cname] = sp["prior_prob"] * np.exp(10 * (c - 0.5) / temp)
    z = sum(scores.values())
    return {k: val / z for k, val in scores.items()}


def anchor_at(engine, word, cname, t):
    """The pointer modulates how strongly sphere cname gates its features:
    smooth prior → gate opens  (multipliers →1, the sphere's shape stops mattering: priors collapse)
    sharp  prior → gate shuts  (multipliers → own coefficients, the sphere filters the world).
    Anchor = argmax cosine over vocab under that gated sphere vector (self-excluded, Law II)."""
    a, b = alpha(t), beta(t)
    g = b / (a + b)                                    # ∈(0,1): gate closure by the sharp side
    coeff = engine.concept_spaces[cname]["coefficients"]
    mx = max(coeff.values()) if coeff else 1.0
    names = list(engine.basis_dict.keys())
    def gated(semes):
        v = np.zeros(len(names))
        for bn, coef in semes.items():
            if bn in names:
                mult = (1 - g) + g * (coeff.get(bn, 1.0) / mx)
                v[names.index(bn)] = coef * mult
        return v
    v_in = engine.encode_word(word)
    nin = np.linalg.norm(v_in)
    best, best_s = None, -2
    for wd, semes in engine.vocab.items():
        if wd == word:
            continue
        vec = gated(semes)
        n = np.linalg.norm(vec)
        if n == 0:
            continue
        sc = float(np.dot(v_in, vec) / (n * nin + 1e-8))
        if sc > best_s:
            best_s, best = sc, wd
    return best, round(best_s, 4)


def main():
    e = SemeBasedCognitiveEngine()
    t = np.linspace(0, 4 * np.pi, 192)
    rep = {"evidence": EV, "curves": {}, "summary": {}}
    for wd in EV:
        H, argmax, flips = [], [], []
        for x in t:
            p = posterior(e, wd, x)
            H.append(-sum(v * np.log(v + 1e-12) for v in p.values()))
            argmax.append(max(p, key=p.get))
        flips = int(sum(1 for i in range(1, len(argmax)) if argmax[i] != argmax[i - 1]))
        rep["curves"][wd] = {"entropy": [round(h, 4) for h in H],
                             "argmax_first": argmax[0], "argmax_last": argmax[-1], "argmax_flips": flips}
        rep["summary"][wd] = {"entropy_min": round(min(H), 3), "entropy_max": round(max(H), 3),
                              "breath_ratio": round(max(H) / min(H), 2), "argmax_flips": flips,
                              "corr_entropy_prior": round(float(np.corrcoef(H, np.array([alpha(x) for x in t]))[0, 1]), 3)}
    # anchor mobility at prior extremes for one witness word
    t_smooth, t_sharp = float(t[int(np.argmax([alpha(x) for x in t]))]), float(t[int(np.argmax([beta(x) for x in t]))])
    rep["anchor"] = {}
    for cname in ["超市", "路边摊"]:
        a1, s1 = anchor_at(e, "塑胶凳", cname, t_smooth)
        a2, s2 = anchor_at(e, "塑胶凳", cname, t_sharp)
        rep["anchor"][cname] = {"smooth_anchor": a1, "sharp_anchor": a2, "moved": a1 != a2,
                                "sim_smooth": s1, "sim_sharp": s2}
    # verdicts (registered in the run-note, threshold 0.2 for breath_ratio spread etc.)
    brs = [s["breath_ratio"] for s in rep["summary"].values()]
    rep["verdicts"] = {
        "Q1_entropy_breathes": bool(min(brs) > 1.2),
        "Q2_argmax_flips": bool(any(s["argmax_flips"] > 0 for s in rep["summary"].values())),
        "Q3_anchor_moves": bool(any(v["moved"] for v in rep["anchor"].values())),
    }
    json.dump(rep, open(os.path.join(OUT, "report_c3_breath.json"), "w"), ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for wd in EV:
        axes[0].plot(t, rep["curves"][wd]["entropy"], label=wd, lw=1.6)
    axes[0].axvline(t_smooth, color="#c9a959", ls=":", label="smooth peak")
    axes[0].axvline(t_sharp, color="#D4763A", ls=":", label="sharp peak")
    axes[0].set_title("Q1 · posterior entropy breathes with the pointer")
    axes[0].set_xlabel("t"); axes[0].legend(fontsize=7)
    p0 = [posterior(e, "塑胶凳", x) for x in t]
    spheres = list(p0[0].keys())
    for sname in spheres:
        axes[1].plot(t, [p[sname] for p in p0], lw=1.4, label=sname)
    axes[1].set_title("Q2 · p(sphere | 塑胶凳, t) — flips visible where curves cross")
    axes[1].set_xlabel("t"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_c3_breath.png"), dpi=150)
    print(json.dumps(rep["verdicts"], indent=1))
    print("anchor:", json.dumps(rep["anchor"], ensure_ascii=False))


if __name__ == "__main__":
    main()
