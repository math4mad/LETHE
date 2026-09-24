#!/usr/bin/env python3
# LADDER-1 干跑臂 · 泛灵驼峰 —— 判据见 PREREG_LADDER-1_anima_hump.md (先于此文件冻结)
import numpy as np, json, itertools, os

DIMS = ["self_propelled","eat","grow","touch","count","static","night","face","human_rel","sound"]
IX = {d:i for i,d in enumerate(DIMS)}
def V(**kw):
    v = np.zeros(len(DIMS))
    for k,x in kw.items(): v[IX[k]] = x
    return v
def cos(a,b):
    na,nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a@b/(na*nb)) if na>1e-9 and nb>1e-9 else 0.0

TAU_A, TAU_B, ALPHA, KAPPA, BETA = 0.50, 0.35, 0.30, 3.0, 6.0

# ── 观测类型 (word, vector, layer) ──────────────────────────────
def obs_stream():
    S=[]
    # L0 玩伴流 (DK L0 层书名钉底) ×3
    L0=[("ball",V(self_propelled=0.0,touch=1.0,count=0.7)),
        ("teddy",V(touch=1.0,face=0.5,static=0.3)),
        ("peekaboo",V(human_rel=0.9,face=0.5,sound=0.6)),
        ("tap",V(touch=1.0,sound=0.7)),
        ("splash",V(self_propelled=0.5,sound=0.8)),
        ("rattle",V(touch=0.8,sound=0.9,count=0.4))]
    for w,v in L0: S+= [(w,v,"L0")]*3
    # L1 生命流 (DK 命名层 + Mother Goose) ×3
    an=[("doggy",V(self_propelled=0.9,eat=0.8,grow=0.7,face=0.6,sound=0.6)),
        ("kitty",V(self_propelled=0.9,eat=0.7,grow=0.7,face=0.6)),
        ("birdie",V(self_propelled=0.9,eat=0.6,grow=0.6,sound=0.7,face=0.3)),
        ("fishy",V(self_propelled=0.8,eat=0.6,grow=0.6)),
        ("baby",V(self_propelled=0.7,grow=0.9,touch=0.8,face=0.8,human_rel=0.9)),
        ("mummy",V(self_propelled=0.8,face=0.8,human_rel=1.0,eat=0.4)),
        ("duckie",V(self_propelled=0.8,eat=0.5,sound=0.6,face=0.3))]
    for w,v in an: S+=[(w,v,"L1")]*3
    # 月亮 L1 泛灵证据 ×5: 「月亮走我也走」+「月亮婆婆」+ Good Night, Baby Moon(在库)
    moon_walk=V(self_propelled=0.9,night=0.8,face=0.6,human_rel=0.5)
    S+=[("moon",moon_walk,"L1")]*5
    # L2 天文/自然流: sun/star/planet/earth/mountain/stone/tree ×3
    L2=[("sun",V(static=0.8,night=-0.6,self_propelled=0.3)),
        ("star",V(static=0.8,night=0.9,self_propelled=0.2,count=0.5)),
        ("planet",V(static=0.9,night=0.5,self_propelled=0.3,count=0.4)),
        ("earth",V(static=0.9,night=0.3)),
        ("mountain",V(static=1.0,grow=0.2)),
        ("stone",V(static=1.0,touch=0.4)),
        ("tree",V(static=0.7,grow=0.8,touch=0.4))]
    for w,v in L2: S+=[(w,v,"L2")]*3
    # 月亮 L2 去泛灵证据 ×5: Sun and Moon(在库) + 绕地转 + 不吃饭不长身体(负系数)
    moon_orbit=V(self_propelled=0.2,static=0.8,night=0.8,eat=-0.5,grow=-0.5)
    S+=[("moon",moon_orbit,"L2")]*5
    return S
REPLAY = [("moon",V(self_propelled=0.95,night=0.8,face=0.6,human_rel=0.5),"replay")]*12  # 回灌×12「月亮走」

# ── 引擎 ────────────────────────────────────────────────────────
class Engine:
    def __init__(self):
        self.wstate={}                      # word -> 特征向量(滑动平均)
        self.spheres={}                     # name -> set(words)
        self.prior={}                       # name -> p
        self.log=[]
    def centroids(self):
        return {n:np.mean([self.wstate[w] for w in ms if w in self.wstate],axis=0)
                for n,ms in self.spheres.items() if ms}
    def assign(self, word, vec):
        cen=self.centroids()
        best,bn=(0.0,None)
        for n,c in cen.items():
            s=cos(vec,c)
            if s>best: best,bn=s,n
        if word in self.spheres.get(bn or "",()): pass
        if best < TAU_B:                                    # 生子球
            nm=f"球·{word}"; self.spheres[nm]=set(); self.prior[nm]=0.05; bn=nm
        self.spheres[bn].add(word); return bn
    def reassess(self):
        cen=self.centroids()
        for w in list(itertools.chain.from_iterable(self.spheres.values())):
            v=self.wstate.get(w)
            if v is None: continue
            scores={n:cos(v,c) for n,c in cen.items()}
            bn=max(scores,key=scores.get); bs=scores[bn]
            cur=[n for n,ms in self.spheres.items() if w in ms][0]
            if bn!=cur and bs>=TAU_A:                       # 迁移
                self.spheres[cur].discard(w); self.spheres[bn].add(w)
    def observe(self, word, vec):
        self.wstate[word] = vec.copy() if word not in self.wstate else (1-ALPHA)*self.wstate[word]+ALPHA*vec
        bn=self.assign(word,vec); self.reassess()
        cen=self.centroids()
        for n,c in cen.items():
            self.prior[n]=self.prior.get(n,0.1)*np.exp(KAPPA*(cos(self.wstate[word],c)-0.4))
        tot=sum(self.prior.values()); self.prior={n:p/tot for n,p in self.prior.items()}
    def membership(self, word):
        if word not in self.wstate or not self.spheres: return {}
        cen=self.centroids(); v=self.wstate[word]
        sims=np.array([cos(v,c) for c in cen.values()]); e=np.exp(BETA*(sims-sims.max()))
        return {n:float(e[i]/e.sum()) for i,(n,_) in enumerate(cen.items())}
    def life_sphere(self):
        for n,ms in self.spheres.items():
            if {"doggy","kitty","birdie"} & ms: return n
        return None
    def has_word_sphere(self,w):
        return [n for n,ms in self.spheres.items() if w in ms]

def snap(eng,tag):
    m=eng.membership("moon"); ls=eng.life_sphere()
    row=dict(checkpoint=tag, moon_members=eng.has_word_sphere("moon"),
             moon_m_life=(m.get(ls) if ls else None),
             priors={k:round(v,3) for k,v in sorted(eng.prior.items(),key=lambda x:-x[1])})
    eng.log.append(row); return row

def run_S():
    eng=Engine()
    stream=obs_stream()
    marks={}
    for i,(w,v,layer) in enumerate(stream):
        eng.observe(w,v)
        marks.setdefault(layer,i)
    out={}
    out["L0末"]=snap(eng,"L0末")  # 重放构建快照: 用分层截断更干净 ↓ 重跑一遍分层
    eng2=Engine(); rows=[]
    stream=obs_stream(); bounds={"L0末":marks["L1"],"L1末":marks["L2"],"L2末":len(stream)}
    j=0
    for tag,stop in bounds.items():
        while j<stop: eng2.observe(*stream[j][:2]); j+=1
        rows.append(snap(eng2,tag))
    life_p_at_L2 = None
    for tag in rows:
        pass
    # 回灌
    for w,v,_ in REPLAY: eng2.observe(w,v)
    rows.append(snap(eng2,"回灌12×后"))
    nat=[n for n,ms in eng2.spheres.items() if {"star","sun","mountain","stone"} & ms]
    fin=dict(rows=rows,
             spheres={n:sorted(ms) for n,ms in eng2.spheres.items()},
             nature_prior_end=sum(eng2.prior.get(n,0) for n in nat),
             nature_p_at_L2end=[r for r in rows if r["checkpoint"]=="L2末"][0]["priors"],
             wobble=max([r["moon_m_life"] or 0 for r in rows if r["checkpoint"]=="回灌12×后"]+[0]))
    return eng2, rows, fin

def run_M(seed):
    rng=np.random.default_rng(seed); S=obs_stream()+REPLAY; rng.shuffle(S)
    eng=Engine(); hump_steps=0; moon_steps=0
    for w,v,_ in S:
        eng.observe(w,v)
        ls=eng.life_sphere()
        if ls and "moon" in eng.wstate:
            moon_steps+=1
            if (eng.membership("moon").get(ls) or 0)>=0.5: hump_steps+=1
    return (hump_steps/moon_steps if moon_steps else 0), len(S)

if __name__=="__main__":
    eng2,rows,fin=run_S()
    print("── S 臂 (逐层喂) ──")
    for r in rows:
        print(f" {r['checkpoint']}: 月亮所在球={r['moon_members']}  m(生命球)={None if r['moon_m_life'] is None else round(r['moon_m_life'],3)}")
    print(" 终态球:", {n:ms for n,ms in fin['spheres'].items()})
    print(" 先验(L2末):", fin['nature_p_at_L2end'])
    print(" 先验(回灌后·自然系合计):", round(fin['nature_prior_end'],3))
    print("── M 臂 (混喂×20种子) ──")
    humps=[run_M(s)[0] for s in range(20)]
    bad=sum(1 for h in humps if h>=0.25)
    print(" 月亮∈生命球占比≥25%流长的种子:", bad, "/20  各种子:", [round(h,2) for h in humps])
    os.makedirs('../analysis',exist_ok=True)
    json.dump(dict(S=rows,M=dict(hump_ratios=humps,seeds_over_25pct=bad),final=fin['spheres']),
              open('../analysis/ladder1_results.json','w'),ensure_ascii=False,indent=1,default=float)
    print("\n[判据] H1 驼峰:", "过" if (rows[1]['moon_m_life'] or 0)>=0.5 and (rows[2]['moon_m_life'] if rows[2]['moon_m_life'] is not None else 1)<=0.2 else "不过",
          "| H3 无驼峰:", "过" if bad<=3 else "不过")
