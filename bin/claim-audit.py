#!/usr/bin/env python3
"""claim-audit.py — 「抽查三类」裸探针（回源验一条）。

一事一验：从台账的「完成／通过／已修复」三类**声明**中各抽 N 条，
回其【源】（文件 / git rev）核对**是否存在**。**不读投影，只碰源**。
全族共塌：任一抽样对不上 → 「台账可信」当场存疑（裸探针律）。

用法:
  python3 claim-audit.py [--n 1] [--claims bin/claims.csv] [--base ~/Programming/code-2026]
退出码: 0=全过; 1=有 FAIL（存疑）。
"""
import csv, os, subprocess, sys, random, argparse, datetime


def check(r, base):
    st = (r.get("source_type") or "").strip()
    ref = (r.get("source_ref") or "").strip()
    if st == "file":
        p = os.path.expanduser(ref if os.path.isabs(ref) else os.path.join(base, ref))
        return os.path.exists(p), p
    if st == "git":
        repo, _, rev = ref.partition(":")
        repo = os.path.expanduser(repo if os.path.isabs(repo) else os.path.join(base, repo)) if repo else base
        rev = rev or "HEAD"
        try:
            rc = subprocess.run(["git", "-C", repo, "cat-file", "-e", rev],
                                capture_output=True, timeout=20).returncode
            return rc == 0, f"{repo}@{rev}"
        except Exception as e:
            return False, f"{repo}@{rev} ({e})"
    return False, f"(unknown source_type={st})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--claims", default="bin/claims.csv")
    ap.add_argument("--base", default="~/Programming/code-2026")
    ap.add_argument("--log", default="bin/claim-audit.log", help="追加一行抽查率 (默认 <repo>/bin/claim-audit.log; 空串=不记)")
    a = ap.parse_args()
    base = os.path.expanduser(a.base)
    rows = list(csv.DictReader(open(a.claims, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault((r.get("kind") or "").strip(), []).append(r)
    print(f"[{datetime.datetime.now():%F %T}] 抽查三类 · 每类 N={a.n} · 台账共 {len(rows)} 条声明")
    total = passn = 0
    for kind in ["完成", "通过", "已修复"]:
        pool = by.get(kind, [])
        if not pool:
            print(f"  [{kind}] （无声明）")
            continue
        for r in random.sample(pool, min(a.n, len(pool))):
            ok, src = check(r, base)
            total += 1
            passn += 1 if ok else 0
            print(f"  [{'PASS' if ok else 'FAIL'}] {kind} · {r.get('claim','')}  →  {src}")
    verdict = "✔ 全过" if (total and passn == total) else "✘ 存疑（回源对不上）"
    line = f"{datetime.datetime.now():%F %T}\t{passn}/{total}\t{verdict}"
    print(f"抽查率 = {passn}/{total}  {verdict}")
    log = a.log if a.log != "" else ""
    if log is None:
        log = ""
    if log:
        lp = log if os.path.isabs(log) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), log)
        try:
            with open(lp, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
    return 0 if (total and passn == total) else 1


if __name__ == "__main__":
    sys.exit(main())
