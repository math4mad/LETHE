#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由 run_template.py 生成两臂 run.py (Kaggle 夜航装弹)。

正本声明: 本脚本是「投影生成器」的唯一正本 —— 生成的臂 run.py 是
`run_template.py` + 本地语料正本 (corpus_friends_s1234.json /
friends_s1234_raw.json) 的派生投影, 语料以 gz+b64 内嵌, sha256 取
解压后字节 (与本地正本逐位同源), 反向对撞校验后落盘。

用法:  python3 make_arms.py          (在 benches/transformer_vis/ 下)
退出码: 0 = 两臂生成+编译+对撞全过; 1 = 任一断言失败
"""
import base64, gzip, hashlib, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG = os.path.dirname(os.path.abspath(__file__))

ARMS = {
    "friends-seasonal-0-5b": dict(
        ARM_TAG="0.5b-selftest",
        ARM_MODEL="Qwen2.5-0.5B-Instruct (bf16)",
        MOUNT="/kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
        QUANT="none"),
    "friends-seasonal-7b": dict(
        ARM_TAG="7b",
        ARM_MODEL="Qwen2.5-7B-Instruct (bnb-4bit, compute fp16)",
        MOUNT="/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
        QUANT="4bit"),
}

KAGGLE_SOURCE_CAP = 1024 * 1024  # Kaggle 源 1MB 上限 (05b 首射 400 实测)


def fill(out, key, val):
    ph = f"#__{key}__#"
    assert ph in out, f"placeholder {ph} not found in template"
    return out.replace(ph, val)


def main():
    corpus = open(os.path.join(ROOT, "corpus_friends_s1234.json"), "rb").read()
    raw = open(os.path.join(ROOT, "friends_s1234_raw.json"), "rb").read()

    # 结构断言: raw 必须有 data.S04.lines (流分析正本锚)
    rj = json.loads(raw)
    assert "data" in rj and "S04" in rj["data"] and "lines" in rj["data"]["S04"], \
        "raw structure broken"
    cj = json.loads(corpus)
    assert cj["seasons"] == [1, 2, 3, 4] and len(cj["items"]) == 1200, \
        "corpus structure broken"

    tpl = open(os.path.join(KG, "run_template.py")).read()

    for arm, kw in ARMS.items():
        out = tpl
        for k in ("ARM_TAG", "ARM_MODEL", "MOUNT", "QUANT"):
            out = fill(out, k, kw[k])
        out = fill(out, "CORPUS_B64", base64.b64encode(gzip.compress(corpus, 9)).decode())
        out = fill(out, "CORPUS_SHA", hashlib.sha256(corpus).hexdigest())
        out = fill(out, "RAW_B64", base64.b64encode(gzip.compress(raw, 9)).decode())
        out = fill(out, "RAW_SHA", hashlib.sha256(raw).hexdigest())
        assert "#__" not in out, f"unfilled placeholder in {arm}"
        assert len(out) < KAGGLE_SOURCE_CAP, \
            f"{arm}: {len(out)} bytes over Kaggle 1MB source cap"

        # 反向对撞: 从生成物解出语料 (gzip 层), 与正本 sha256 逐位同源
        m = re.search(r'_CORPUS_B64 = "([^"]+)"', out)
        back = gzip.decompress(base64.b64decode(m.group(1)))
        assert hashlib.sha256(back).hexdigest() == hashlib.sha256(corpus).hexdigest(), \
            f"{arm} corpus roundtrip FAIL"
        m2 = re.search(r'_RAW_B64 = "([^"]+)"', out)
        back2 = gzip.decompress(base64.b64decode(m2.group(1)))
        assert hashlib.sha256(back2).hexdigest() == hashlib.sha256(raw).hexdigest(), \
            f"{arm} raw roundtrip FAIL"

        path = os.path.join(KG, arm, "run.py")
        open(path, "w").write(out)
        print(f"{arm}: run.py {len(out)/1e6:.3f} MB (cap 1.000) · corpus+raw roundtrip sha256 OK")

        r = subprocess.run([sys.executable, "-m", "py_compile", path],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{arm} py_compile FAIL:\n{r.stderr}"
        print(f"{arm}: py_compile OK")
    print("GENERATION_DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print("MAKE_ARMS_FAIL:", e)
        sys.exit(1)
