#!/usr/bin/env python3
"""zsim (Zero 整数定点) ↔ cognitive_engine.py (Python 浮点) 差分比对台。

用途：用两套相互独立的数值实现跑同一套义素基，逐项对表 —— **差异即信号**。
这两个引擎的「内部词」归属是两条独立路径：
  - Python 版：显式声明的 concept_spaces[...]["members"]
  - Zero 版  ：argmax_c cos(词, c) 算法推导
若两者漂移、或后验/ MaxSim 数值超出门限，本脚本以非零退出码报错。

用法:
    python3 zsim/compare.py [词 ...]        # 默认跑内置样本
    python3 zsim/compare.py --quiet         # 只报汇总，不印全表
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))      # <repo>/zsim
GARDEN = os.path.dirname(HERE)                          # <repo>
TOL_MAXSIM = 0.01     # 浮点 vs 整数定点的容许差
TOL_POST_BP = 5       # 后验容许差（basis point）

RE_PY_MAX = re.compile(r"与 \[(.+?)\] 的最大相似度: ([-\d.]+) \(由内部词 '(.+?)' 贡献\)")
RE_PY_NONE = re.compile(r"与 \[(.+?)\] 无独立锚点")
RE_PY_POST = re.compile(r"^\s+(\S+)\s+\|\s+([0-9.]+)\s+\|", re.M)
RE_Z_MAX = re.compile(r"^\s+(\S+)\s+maxSim=(-?\d+) bp\s+锚=(\S*)\s*$", re.M)
RE_Z_NONE = re.compile(r"^\s+(\S+)\s+maxSim=n/a\s+锚=\(无独立锚点\)", re.M)
RE_Z_POST = re.compile(r"^\s+(\S+)\s+(-?\d+) bp\s", re.M)


def find_python_with_numpy():
    """找一个能 import numpy 的解释器（本机 /usr/bin/python3 常没有）。"""
    cands = [os.environ.get("PYTHON"), "/usr/local/bin/python3",
             "/opt/homebrew/bin/python3", shutil.which("python3"), sys.executable]
    for c in cands:
        if not c or not os.path.exists(c):
            continue
        r = subprocess.run([c, "-c", "import numpy"], capture_output=True)
        if r.returncode == 0:
            return c
    sys.exit("compare: 找不到带 numpy 的 python3（设环境变量 PYTHON 指定）")


def find_zero():
    """找一个 zero 编译器：环境变量 ZERO > 仓内 .bin > 邻域沙盒 > PATH。"""
    cands = [os.environ.get("ZERO"),
             os.path.join(GARDEN, ".bin", "zero"),
             os.path.expanduser("~/Programming/code-2026/zero-sandbox/.bin/zero"),
             shutil.which("zero")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    sys.exit("compare: 找不到 zero 编译器；见 zsim/README.md 的安装说明")


PY = find_python_with_numpy()
ZERO = find_zero()


def run_python(word):
    r = subprocess.run([PY, "cognitive_engine.py", word], cwd=GARDEN,
                       capture_output=True, text=True)
    out = r.stdout
    mx = {m[0]: (float(m[1]), m[2]) for m in RE_PY_MAX.findall(out)}
    for m in RE_PY_NONE.findall(out):
        mx[m] = (None, None)
    post = {m[0]: int(round(float(m[1]) * 10000)) for m in RE_PY_POST.findall(out)}
    return mx, post


def run_zsim(word):
    r = subprocess.run([ZERO, "run", "--", word], cwd=HERE,
                       capture_output=True, text=True)
    out = r.stdout
    mx = {m[0]: (int(m[1]), m[2]) for m in RE_Z_MAX.findall(out)}
    for m in RE_Z_NONE.findall(out):
        mx[m] = (None, None)
    post = {m[0]: int(m[1]) for m in RE_Z_POST.findall(out)}
    return mx, post


def compare_word(word, quiet=False):
    pmx, ppost = run_python(word)
    zmx, zpost = run_zsim(word)
    bad = 0
    if not quiet:
        print("=" * 78)
        print("Evidence: 【%s】" % word)
        print("-" * 78)
        print("%-10s | %-20s | %-20s | %s" % ("空间", "Python MaxSim", "zsim MaxSim", "一致?"))
        print("-" * 78)
        for k in pmx:
            pv, pa = pmx[k]
            zv, za = zmx.get(k, (None, "?"))
            zpv = None if zv is None else zv / 10000.0
            if pv is None and zv is None:
                ok = True
            elif pv is not None and zpv is not None:
                ok = abs(pv - zpv) < TOL_MAXSIM
            else:
                ok = False
            bad += 0 if ok else 1
            print("%-10s | %s (%-10s) | %s (%-10s) | %s" % (
                k, "无" if pv is None else "%.4f" % pv, (pa or "—")[:10],
                "无" if zpv is None else "%.4f" % zpv, (za or "—")[:10],
                "  ✓" if ok else "  ✗ 差"))
        print("-" * 78)
        print("%-10s | %10s | %10s | %s" % ("后验(bp)", "Python", "zsim", "一致?"))
        print("-" * 78)
        for k in ppost:
            p, z = ppost[k], zpost.get(k)
            d = abs(p - z) if z is not None else None
            ok = d is not None and d <= TOL_POST_BP
            bad += 0 if ok else 1
            print("%-10s | %10d | %10s | %s" % (k, p, z if z is not None else "n/a",
                                                "  ✓" if ok else "  ✗ %s" % d))
        print()
    else:
        for k in pmx:
            pv, _ = pmx[k]
            zv, _ = zmx.get(k, (None, None))
            if pv is None and zv is None:
                continue
            if pv is None or zv is None or abs(pv - zv / 10000.0) >= TOL_MAXSIM:
                bad += 1
        for k in ppost:
            z = zpost.get(k)
            if z is None or abs(ppost[k] - z) > TOL_POST_BP:
                bad += 1
    return bad


def main(argv):
    quiet = "--quiet" in argv
    words = [a for a in argv if not a.startswith("--")] or [
        "塑胶凳", "煤气罐", "沃尔玛", "收银台", "烧烤摊",
        "我不要再被人摆布", "一切都按计划进行，分秒不差"]
    print("python: %s" % PY)
    print("zero  : %s" % ZERO)
    print()
    bad = sum(compare_word(w, quiet) for w in words)
    print("=" * 78)
    if bad:
        print("✗ 差分验证未过：%d 处不一致（%d 词）" % (bad, len(words)))
        return 1
    print("✓ 差分验证通过：%d 词逐项一致（MaxSim 容差 %.2f，后验容差 %d bp）"
          % (len(words), TOL_MAXSIM, TOL_POST_BP))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
