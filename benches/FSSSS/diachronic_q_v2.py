# -*- coding: utf-8 -*-
"""
历时提问测试 v2 — 真模型（黑箱 LM）版
=======================================
被测从"理想化系统"(v0/v1) 换成**真黑箱语言模型**：能否答出阶段 A→B 的结构变化，并守边界（iCar 不出现）。
判据先冻：见 DESIGN_diachronic_q_v2.md（H-v2-1..H-v2-4）。

后端: --backend stub|llama|openai
  stub   内建行为档案(oracle/flattener/confabulator) → 仪器自检(无模型可跑)
  llama  llama-cli -m <gguf>（待供模型）
  openai OpenAI 兼容端点（待供端点）

运行:
  python3 benches/FSSSS/diachronic_q_v2.py --backend stub --profile all
  python3 benches/FSSSS/diachronic_q_v2.py --backend llama --model /path/to.gguf
"""
import os
import re
import json
import argparse
import subprocess

GT_NEW = {"iPhone", "iPad", "iPod"}      # A→B 应出现的同域兄弟
SEED = "iMac"
NEG = "iCar"                             # 越界反事实负例

PROMPT = """You are given two stages of a product concept space.

Stage A:
- base concepts: Mac, Phone, Pad, Pod (domain: consumer electronics), Car (domain: vehicle)
- a seed compound has appeared: iMac (formed as "i" + "Mac")

Stage B:
- the "i-" compound family extends by analogy from the seed, staying inside the same domain
  as the seed.

Question:
(1) Which i-compounds appear in Stage B but not in Stage A?
(2) Does "iCar" appear?

Answer STRICTLY in this format, nothing else:
APPEARS: <comma-separated names, or none>
ICAR: yes|no
"""


# ── 后端 ────────────────────────────────────────────────────────────────
def call_stub(profile):
    return {
        "oracle": "APPEARS: iPhone, iPad, iPod\nICAR: no",
        "flattener": "APPEARS: none\nICAR: no",
        "confabulator": "APPEARS: iPhone, iPad, iPod, iCar\nICAR: yes",
    }[profile]


def call_llama(model, prompt, n=192):
    try:
        r = subprocess.run(["llama-cli", "-m", model, "-p", prompt, "-n", str(n),
                            "--temp", "0", "-no-cnv"],
                           capture_output=True, text=True, timeout=300)
        return (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception as e:  # noqa
        return f"[llama error] {e}"


def call_openai(endpoint, model, prompt, key_env="OPENAI_API_KEY"):
    import urllib.request
    key = os.environ.get(key_env, "")
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"]["content"]
    except Exception as e:  # noqa
        return f"[openai error] {e}"


# ── 解析 / 计分 ─────────────────────────────────────────────────────────
def parse_answer(text):
    m_app = re.search(r"APPEARS\s*:\s*(.*)", text, re.I)
    m_icar = re.search(r"ICAR\s*:\s*(yes|no)", text, re.I)
    raw = (m_app.group(1).strip() if m_app else "")
    appears = set()
    if raw and raw.lower() not in ("none", "n/a", "-"):
        appears = {x.strip() for x in re.split(r"[,\n]", raw) if x.strip()}
    icar = None
    if m_icar:
        icar = m_icar.group(1).lower() == "yes"
    return appears, icar


def score(appears, icar):
    recall = len(GT_NEW & appears) / len(GT_NEW)
    allowed = GT_NEW | {SEED}
    false_new = len({a for a in appears if a not in allowed})
    icar_correct = (icar is False) and (NEG not in appears)
    return {"structural_recall": recall, "icar_correct": icar_correct, "false_new": false_new}


def verdict(sc):
    h1 = sc["structural_recall"] == 1.0
    h2 = sc["icar_correct"]
    h3 = sc["false_new"] == 0
    return {"H-v2-1": h1, "H-v2-2": h2, "H-v2-3": h3}


def run_one(name, text):
    appears, icar = parse_answer(text)
    sc = score(appears, icar)
    v = verdict(sc)
    print(f"\n── {name} ──")
    print(f"   raw: {text.strip()[:200]!r}")
    print(f"   解析: appears={sorted(appears)}  icar={icar}")
    print(f"   计分: 召回={sc['structural_recall']:.3f}  守边界={sc['icar_correct']}  真值外节点={sc['false_new']}")
    print(f"   判据: " + "  ".join(f"{k}={'✓' if x else '✗'}" for k, x in v.items()))
    return {"appears": sorted(appears), "icar": icar, "score": sc, "verdict": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="stub", choices=["stub", "llama", "openai"])
    ap.add_argument("--profile", default="all", choices=["all", "oracle", "flattener", "confabulator"])
    ap.add_argument("--model", default="")
    ap.add_argument("--endpoint", default="http://127.0.0.1:1234")
    ap.add_argument("--model-name", default="local")
    args = ap.parse_args()

    print("=" * 88)
    print("历时提问测试 v2 — 真模型（黑箱 LM）版")
    print("=" * 88)
    print(f"真值: A→B 应出现 {sorted(GT_NEW)} ; 越界负例 {NEG} 应 ICAR=no")

    results = {}
    if args.backend == "stub":
        profiles = ["oracle", "flattener", "confabulator"] if args.profile == "all" else [args.profile]
        for p in profiles:
            results[p] = run_one(f"stub[{p}]", call_stub(p))
    elif args.backend == "llama":
        if not args.model:
            print("\n[需 --model <gguf>] 本机无 GGUF；真模型启用条件见 DESIGN_diachronic_q_v2.md §6")
        else:
            results["llama"] = run_one("llama-cli", call_llama(args.model, PROMPT))
    else:
        results["openai"] = run_one("openai-endpoint", call_openai(args.endpoint, args.model_name, PROMPT))

    # H-v2-4 仪器判别力（stub 三档判词互异）
    H_v2_4 = None
    if args.backend == "stub" and len(results) >= 2:
        sigs = {k: tuple(sorted(v["verdict"].items())) for k, v in results.items()}
        H_v2_4 = len(set(sigs.values())) == len(sigs)
        print(f"\nH-v2-4 仪器判别力（stub 各档判词互异）: {H_v2_4}  "
              f"{'✓' if H_v2_4 else '✗'}  （{len(sigs)} 档 → {len(set(sigs.values()))} 种判词）")

    out = {"backend": args.backend, "ground_truth": {"appears": sorted(GT_NEW), "icar": False},
           "results": results, "H-v2-4": H_v2_4,
           "real_model_available": False,
           "enable_condition": "供 GGUF 或 OpenAI 兼容端点，或上 Kaggle（见 DESIGN §6）"}
    path = os.path.join(os.path.dirname(__file__), "report_diachronic_q_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
