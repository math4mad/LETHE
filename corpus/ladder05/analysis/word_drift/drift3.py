#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""drift3.py — 词义漂移 v2（**修正 v0 的循环**：探针集排除目标词自身；并报孤立率）

v0 的两处病（本次修正）：
 ① 循环：目标词 dog∈ANIM、ball∈OBJ —— "同现率"拿自己证自己（dog 恒 +1、ball 恒 −1 全是假的）。
 ② 孤立：moon 71–94% 的话轮里除它自己无别的实词 → y≈0 是**测量空洞**，非语义居中。
修正：anim/obj 判定一律 **排除目标词自身**；另报孤立率，孤立率高者结果不作数。
预注册不变：P1 moon 泛灵坐标随龄下降；P2/P3 锚 dog·baby 恒高 / ball·car·book 恒低。
"""
import os, re, glob, json, collections

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.join(HERE, "..", "..", "layers", "childes", "chat")
ANIM = set("""mommy mama mummy daddy dada baby dog doggie puppy kitty cat bird fish
boy girl man lady people person guy duck bunny rabbit horse cow pig bear""".split())
OBJ = set("""ball block cup shoe book car truck milk water juice bottle box
chair table spoon fork paper crayon pencil flower tree rock stone""".split())
TARGETS = ["moon","sun","star","light","flower","dog","baby","kitty","ball","car","book","milk",
           "cup","tree","car"]
TARGETS = list(dict.fromkeys(TARGETS))
BANDS = [(13,23),(24,33),(34,43),(44,53),(54,62)]
BANDLAB = [f"{a}–{b}" for a,b in BANDS]

def age(t):
    m=re.search(r'^@ID:\s*eng\|[^|]*\|CHI\|(\d+);(\d+)\.',t,re.M)
    return None if not m else int(m.group(1))*12+int(m.group(2))
def band(x):
    for i,(a,b) in enumerate(BANDS):
        if a<=x<=b: return i
    return None
def words(line):
    out=[]
    for raw in line[5:].strip().split():
        x=raw.lower(); x=re.sub(r'\(.*?\)','',x); x=x.replace("'s","").replace("'","")
        x=re.sub(r'[^a-z]+','',x)
        if x: out.append(x)
    return out

acc=collections.defaultdict(lambda:[0,0,0,0])   # (b,t)->[n, anim, obj, iso]
nsess=collections.Counter()
for f in sorted(glob.glob(os.path.join(CHAT,"*.cha"))):
    txt=open(f,encoding='utf-8',errors='ignore').read(); m=age(txt)
    if m is None: continue
    b=band(m)
    if b is None: continue
    nsess[b]+=1
    for line in txt.splitlines():
        if not line.startswith("*CHI:"): continue
        W=words(line)
        for t in set(W)&set(TARGETS):
            others=set(W)-{t}
            a=bool(others&ANIM); o=bool(others&OBJ)
            r=acc[(b,t)]; r[0]+=1; r[1]+=a; r[2]+=o; r[3]+= (not a and not o)
table={}
for t in TARGETS:
    row=[]
    for b in range(len(BANDS)):
        n,an,ob,iso=acc.get((b,t),[0,0,0,0])
        row.append(None if n==0 else dict(n=n, anim=round(an/n,2), obj=round(ob/n,2),
                                          y=round((an-ob)/n,2), iso=round(iso/n,2)))
    table[t]=row
json.dump(dict(bands=BANDS,bandlab=BANDLAB,table=table,
               sessions={BANDLAB[b]:nsess[b] for b in range(len(BANDS))}),
          open(os.path.join(HERE,"drift3.json"),"w"),ensure_ascii=False,indent=1)
L=["# 词义漂移 v2（探针排除自身 · 带孤立率）","",
   "| 词 | "+" | ".join(BANDLAB)+" |","|---|"+"---|"*len(BANDLAB)]
for t in TARGETS:
    cs=[]
    for r in table[t]:
        cs.append("—" if r is None else (f"**{r['y']:+.2f}** 孤{r['iso']:.0%} (n={r['n']})"))
    L.append(f"| {t} | "+" | ".join(cs)+" |")
L+=["","会话/带: "+" · ".join(f"{BANDLAB[b]}={nsess[b]}" for b in range(len(BANDS)))]
md="\n".join(L); open(os.path.join(HERE,"table3.md"),"w").write(md+"\n"); print(md)
print("\nDRIFT3_DONE")
