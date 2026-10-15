#!/usr/bin/env python3
"""构建 Geometry-of-Meaning 7B Kaggle 夜航臂 (geometry-meaning-7b)

三样交付律:
  本生成器 = 正本 (臂的唯一来源)
  geometry-meaning-7b/run.py = 投影 (由本脚本生成, 禁手改)
  verify = verify_geometry_kaggle.py (回收后对表)

正本语料: ~/CodeBuddy/Geometry-Of-Meaning/corpus_paraphrase.json
模板:     ./run_geometry_template.py

用法: python3 make_geometry_arm.py
"""
import base64
import gzip
import hashlib
import os
import py_compile
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_SRC = os.path.expanduser("~/CodeBuddy/Geometry-Of-Meaning/corpus_paraphrase.json")
TEMPLATE = os.path.join(HERE, "run_geometry_template.py")
ARM_DIR = os.path.join(HERE, "geometry-meaning-7b")

MOUNT = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
QUANT = "4bit"
TAG = "geometry-meaning-7b"
MODEL_DESC = "Qwen2.5-7B-Instruct (bnb-4bit, 挂载 qwen-lm)"


def main():
    corpus = open(CORPUS_SRC, "rb").read()
    corpus_sha = hashlib.sha256(corpus).hexdigest()
    corpus_b64 = base64.b64encode(gzip.compress(corpus, compresslevel=9)).decode()
    print(f"正本语料: {CORPUS_SRC}")
    print(f"  bytes={len(corpus)}  sha256[:16]={corpus_sha[:16]}")
    print(f"  gz+b64={len(corpus_b64)} chars (Kaggle 源上限 1MB)")

    tpl = open(TEMPLATE, encoding="utf-8").read()
    arm = (tpl.replace("#__ARM_TAG__#", TAG)
              .replace("#__ARM_MODEL__#", MODEL_DESC)
              .replace("#__MOUNT__#", MOUNT)
              .replace("#__QUANT__#", QUANT)
              .replace("#__CORPUS_B64__#", corpus_b64)
              .replace("#__CORPUS_SHA__#", corpus_sha))
    os.makedirs(ARM_DIR, exist_ok=True)
    arm_path = os.path.join(ARM_DIR, "run.py")
    with open(arm_path, "w", encoding="utf-8") as f:
        f.write(arm)
    py_compile.compile(arm_path, doraise=True)
    print(f"生成: {arm_path} (py_compile OK)")

    # 对撞: 生成物内嵌语料解压后 sha == 模板 sha == 正本 sha
    src = open(arm_path, encoding="utf-8").read()
    m_b64 = re.search(r'_CORPUS_B64 = "([^"]+)"', src)
    m_sha = re.search(r'_CORPUS_SHA = "([^"]+)"', src)
    assert m_b64 and m_sha, "占位符替换不完全"
    data = gzip.decompress(base64.b64decode(m_b64.group(1)))
    got = hashlib.sha256(data).hexdigest()
    assert got == m_sha.group(1) == corpus_sha, \
        f"对撞失败: 生成臂 {got} vs 正本 {corpus_sha}"
    print(f"对撞 OK: 生成臂内嵌语料 sha256 == 正本 ({corpus_sha[:16]})")
    print(f"臂目录: {ARM_DIR}/  (run.py + kernel-metadata.json + PREREG.md)")


if __name__ == "__main__":
    main()
