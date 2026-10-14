#!/usr/bin/env python3
"""从 friends_s{seasons}_raw.json 抽样生成多季语料正本。

设计: 每季每角色 PER_ROLE 条, 跨集系统抽样(句序均匀), 长度过滤。
      stage = 角色索引(0-5, 与 corpus_friends.json 同构),
      season = 季号, name = 角色名, desc = 台词原文。
      支持任意季列表 (默认 S01-S04)。

旋钮:
  SEED=13
  PER_ROLE=50
  MIN_WORDS=2, MAX_WORDS=35
  SEASONS=[1,2,3,4]
  ROLES=[Rachel, Chandler, Ross, Monica, Joey, Phoebe] (主角对在前)
"""
import json, random, sys

SEED = 13
PER_ROLE = 50
MIN_WORDS = 2
MAX_WORDS = 35
SEASONS = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4]
ROLES = ['Rachel', 'Chandler', 'Ross', 'Monica', 'Joey', 'Phoebe']

def main():
    raw = json.load(open('friends_s' + ''.join(str(s) for s in SEASONS) + '_raw.json'))
    rng = random.Random(SEED)
    items = []
    stats = {}
    for season in SEASONS:
        sk = f'S{season:02d}'
        lines = raw['data'][sk]['lines']
        for ri, role in enumerate(ROLES):
            pool = [l['line'] for l in lines if l['speaker'] == role]
            pool = [l for l in pool if MIN_WORDS <= len(l.split()) <= MAX_WORDS]
            step = len(pool) / PER_ROLE
            picks = [pool[min(len(pool) - 1, int(i * step))] for i in range(PER_ROLE)]
            for p in picks:
                items.append({'name': role, 'desc': p, 'stage': ri, 'season': season})
            stats[f'S{season:02d}:{role}'] = {'pool': len(pool), 'picked': len(picks)}
    corpus = {
        'deck': 'D5_friends_s' + ''.join(str(s) for s in SEASONS),
        'minted': '2026-10-13',
        'note': (f'Friends S{SEASONS[0]:02d}-S{SEASONS[-1]:02d} 六主角台词语料: '
                 '每季每角色 50 条, 跨集系统抽样。来源 fangj.github.io/friends。'
                 '检验假设: Rachel 的角色特色是否在 S03/S04 才凝聚 (主人观感)。'),
        'roles': ROLES,
        'seasons': SEASONS,
        'sampling': {'seed': SEED, 'per_role': PER_ROLE,
                     'min_words': MIN_WORDS, 'max_words': MAX_WORDS},
        'stats': stats,
        'items': items,
    }
    fname = 'corpus_friends_s' + ''.join(str(s) for s in SEASONS) + '.json'
    json.dump(corpus, open(fname, 'w'), ensure_ascii=False, indent=1)
    print(f"{fname}: {len(items)} 条 ({len(SEASONS)} 季 × 6 角色 × {PER_ROLE})")
    # 每季池大小概览
    for season in SEASONS:
        sk = f'S{season:02d}'
        pools = {r: stats[f'{sk}:{r}']['pool'] for r in ROLES}
        print(f"  {sk}: " + ', '.join(f"{r.split()[0]}={pools[r]}" for r in ROLES))

if __name__ == '__main__':
    main()
