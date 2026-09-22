#!/usr/bin/env python3
# bin/queue.py — 零依赖文件队列 (BullMQ 语义子集, 文件系统实现)
# 园律: 失败必留痕 — 打满者入 dead/ (死信即挂账), 静默消失是重罪。
# 用法:
#   bin/queue.py add <id> --cmd "..." [--tries 6] [--base 300] [--note "..."]
#   bin/queue.py tick                      # 到期任务执行一轮 (launchd 每 5 分钟喂)
#   bin/queue.py list                      # 在队/死信一览
#   bin/queue.py retry <id>                # 死信复活 (补进 .pending 后重入队)
# 退避: next_at = now + base * 2^attempts (封顶 2h) → 300s 粒度下即
#       5,10,20,40,80,160min ≈ 主人重试律的 快试+10/20/40 加宽版。
import json, sys, time, subprocess, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / ".queue"
JOBS, DEAD = Q / "jobs", Q / "dead"
MAX_BACK = 7200

def now(): return int(time.time())

def stamp(): return time.strftime("%F %T")

def jpath(i, d=JOBS): return d / f"{i}.json"

def add(jid, cmd, tries, base, note):
    JOBS.mkdir(parents=True, exist_ok=True); DEAD.mkdir(parents=True, exist_ok=True)
    if jpath(jid).exists():
        print(f"✘ 在队 · {jid} 已有同 id 任务 (先 rm 或换 id)"); sys.exit(1)
    j = dict(id=jid, cmd=cmd, tries=tries, base=base, note=note,
             attempts=0, next_at=now(), created=stamp(), log=[])
    jpath(jid).write_text(json.dumps(j, ensure_ascii=False, indent=1))
    print(f"⏬ 入队 · {jid} · tries={tries} base={base}s · next={stamp()}")

def tick():
    if not JOBS.exists(): print("(队列未启用)"); return
    ran = 0
    for f in sorted(JOBS.glob("*.json")):
        j = json.loads(f.read_text())
        if j["next_at"] > now(): continue
        ran += 1
        try:
            r = subprocess.run(["bash", "-lc", j["cmd"]], capture_output=True, text=True, timeout=240)
            ok, out = r.returncode == 0, (r.stdout + r.stderr).strip()[-400:]
        except subprocess.TimeoutExpired:
            ok, out = False, "TIMEOUT 240s"
        j["attempts"] += 1
        j["log"].append(f"{stamp()} try{j['attempts']} {'ok' if ok else 'fail'} :: {out[:200]}")
        j["log"] = j["log"][-8:]
        if ok:
            (Q / "done").mkdir(exist_ok=True)
            j["done"] = stamp()
            jpath(j["id"], Q / "done").write_text(json.dumps(j, ensure_ascii=False, indent=1))
            f.unlink()
            print(f"☑ {j['id']} · 第{j['attempts']}试成")
        elif j["attempts"] >= j["tries"]:
            j["dead"] = stamp()
            jpath(j["id"], DEAD).write_text(json.dumps(j, ensure_ascii=False, indent=1))
            f.unlink()
            pend = ROOT / ".pending-push"
            line = f"{stamp()} QUEUE-DEAD · {j['id']} · {j['tries']}试皆败 · cmd={j['cmd'][:120]}"
            with open(pend, "a") as pf: pf.write(line + "\n")
            print(f"☠ {j['id']} · 死信挂账 (.pending-push + dead/)")
        else:
            back = min(j["base"] * (2 ** j["attempts"]), MAX_BACK)
            j["next_at"] = now() + back
            f.write_text(json.dumps(j, ensure_ascii=False, indent=1))
            print(f"↻ {j['id']} · 败{j['attempts']}次 · {back//60}分后再试")
    if ran == 0: print(f"· tick @ {stamp()} · 无到期任务")

def lst():
    for d, tag in ((JOBS, "◻ 在队"), (DEAD, "☠ 死信"), (Q / "done", "☑ 成")):
        if not d or not d.exists(): continue
        for f in sorted(d.glob("*.json")):
            j = json.loads(f.read_text())
            nxt = time.strftime("%H:%M", time.localtime(j.get("next_at", 0)))
            print(f"{tag} {j['id']:28s} tries={j.get('attempts',0)}/{j['tries']} next={nxt} {j.get('note','')[:40]}")

def retry(jid):
    src = jpath(jid, DEAD)
    if not src.exists(): print(f"✘ dead/ 无 {jid}"); sys.exit(1)
    j = json.loads(src.read_text()); j["attempts"] = 0; j["next_at"] = now(); j["log"] = []
    src.unlink(); jpath(jid).write_text(json.dumps(j, ensure_ascii=False, indent=1))
    print(f"♻ 复活 · {jid} 重入队")

if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__ or "用法见文件头"); sys.exit(0)
    a = sys.argv[1]
    if a == "add":
        rest = sys.argv[2:]; jid = rest[0]; cmd = ""; tries, base, note = 6, 300, ""
        it = iter(range(1, len(rest)))
        i = 1
        while i < len(rest):
            if rest[i] == "--cmd": cmd = rest[i+1]; i += 2
            elif rest[i] == "--tries": tries = int(rest[i+1]); i += 2
            elif rest[i] == "--base": base = int(rest[i+1]); i += 2
            elif rest[i] == "--note": note = rest[i+1]; i += 2
            else: i += 1
        if not cmd: print("✘ 缺 --cmd"); sys.exit(1)
        add(jid, cmd, tries, base, note)
    elif a == "tick": tick()
    elif a == "list": lst()
    elif a == "retry": retry(sys.argv[2])
    else: print("未知动作: add|tick|list|retry")
