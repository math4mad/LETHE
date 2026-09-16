#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Archiver's鞭 · archive ledger deriver
    board.html 家法循例：意图（本文件的存在）手写，车道派生自字节 —
    从 git log --name-status 与全部 *.lethe-pin.json 收据，
    派生 website/archive.json，供 archiver.html 渲染。

列出：① 每个文件的修改历史版本（A/M/D 诸事件，新者在前）
      ② 所有添加版本（status=A 者，即入册之日）
      ③ 收据簿（每件带条字节：path, sha256, producer, purpose, received）

重跑: python3 bin/archive-to-json.py
"""
import glob
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "website", "archive.json")

SEP = "\x1f"   # unit separator: marks a commit-header line (sha\x1fdate\x1fsubject)


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args],
                          capture_output=True, text=True, check=True).stdout


def parse_log(raw):
    """逐行解析 git log --name-status：含 SEP 者为 commit 头，余者为文件事件。"""
    commits, files = [], {}
    cur = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if SEP in line:
            sha, date, subject = line.split(SEP, 2)
            cur = {"sha": sha, "date": date, "subject": subject.strip(), "files": []}
            commits.append(cur)
            continue
        parts = line.split("\t")
        status, path = parts[0], parts[-1]
        ev = {"status": status[0], "path": path}
        cur["files"].append(ev)
        cur_files = files.setdefault(path, [])
        if not cur_files or cur_files[0]["sha"] != cur["sha"][:7]:
            cur_files.insert(0, {"sha": cur["sha"][:7], "date": cur["date"],
                                 "subject": cur["subject"], "status": status[0]})
    return commits, files


def main():
    head = git("rev-parse", "HEAD").strip()
    remote = ""
    try:
        remote = git("remote", "get-url", "origin").strip()
    except subprocess.CalledProcessError:
        pass

    fmt = SEP.join(["%H", "%ad", "%s"])
    raw = git("log", "--name-status", "--date=iso-strict",
              f"--pretty=format:{fmt}", "--")
    commits, files = parse_log(raw)

    file_rows = sorted(
        ({"path": p, "versions": len(v),
          "added": next((e["date"] for e in v if e["status"] == "A"), None),
          "last": v[0]["date"], "events": v}
         for p, v in files.items()),
        key=lambda r: r["path"])

    receipts = []
    for pin in sorted(glob.glob(os.path.join(ROOT, "**", "*.lethe-pin.json"),
                                recursive=True)):
        rel = os.path.relpath(pin, ROOT)
        try:
            d = json.load(open(pin))
        except (json.JSONDecodeError, OSError):
            d = {"error": "unreadable receipt", "path": rel}
        receipts.append({
            "sidecar": rel,
            "artifact": d.get("path", rel.replace(".lethe-pin.json", "")),
            "sha256": d.get("sha256", ""),
            "producer": d.get("producer", ""),
            "purpose": d.get("purpose", ""),
            "received": d.get("received", ""),
        })

    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "head": head[:7],
        "head_full": head,
        "remote": remote,
        "commits": commits,
        "files": file_rows,
        "receipts": receipts,
        "law": "git history is the receipt for code; pin sidecars are the "
               "receipt for bytes; an Archiver that can be edited in place is "
               "no archive — regenerate or perish.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"⚏ archive.json 已派生 · {len(commits)} commits · "
          f"{len(file_rows)} files · {len(receipts)} receipts → website/archive.json")


if __name__ == "__main__":
    main()
