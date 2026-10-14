#!/usr/bin/env python3
"""从 fangj.github.io/friends 提取多季全六主角台词（说话人标注正本）。

数据源: https://fangj.github.io/friends/season/{SS}{EE}.html (GitHub Pages, 静态 HTML)
格式: <p...><b>Name:</b> 台词</p> —— 完整台词(无挖空)
清洗: 去 HTML 标签 / 舞台说明(括号) / 场景[括号] / 斜体歌声
输出: friends_s{ seasons }_raw.json = {season: {per_episode, lines:[(speaker,line)]}}

旋钮:
  SEASONS=[1,2,3,4]
  SIX={Monica,Chandler,Joey,Phoebe,Ross,Rachel}
"""
import urllib.request, re, json, time, sys
from collections import Counter

SEASONS = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4]
SIX = {'Monica', 'Chandler', 'Joey', 'Phoebe', 'Ross', 'Rachel'}
BASE = 'https://fangj.github.io/friends/season/'

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(req, timeout=30).read().decode('utf-8', errors='replace')

def _clean(ln):
    ln = re.sub(r'<[^>]+>', ' ', ln)
    ln = ln.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&quot;', '"')
    ln = re.sub(r'\([^)]*\)', ' ', ln)   # 台词内舞台说明
    ln = re.sub(r'\[[^\]]*\]', ' ', ln)  # 场景
    ln = re.sub(r'\s+', ' ', ln).strip()
    return ln

def parse_ep(html):
    """兼容两式:
    S01/S03/S04: 每句一个 <p> 块, <b>Name:</b> 台词
    S02: 一个 <p align="left"> 块装整集, 台词行用 <br> 分隔,
         行首 [Scene: ...] 为舞台说明, NAME: 全大写
    """
    lines = []
    for pm in re.finditer(r'<p[^>]*>(.*?)</p>', html, re.S):
        block = pm.group(1)
        # 式1 (S01/S03/S04 标准): <b>Name:</b> 台词
        m = re.search(r'<b>([A-Za-z ]+):</b>\s*', block)
        if m:
            sp = m.group(1).strip()
            ln = _clean(block[m.end():])
            if sp in SIX and len(ln) > 3:
                lines.append({'speaker': sp, 'line': ln})
            continue
        # 式3 (S01E16/S04E06): <p>Name</b>: 或 <p>Name:</b> 台词 (前有游离 <b>)
        m3 = re.match(r'\s*([A-Za-z ]+):?</b>:?\s*', block)
        if m3:
            sp = m3.group(1).strip()
            ln = _clean(block[m3.end():])
            if sp in SIX and len(ln) > 3:
                lines.append({'speaker': sp, 'line': ln})
            continue
        # 式4 (S04E03/E06): <p><strong>Name:</strong> 台词
        m4 = re.match(r'\s*<strong>([A-Za-z ]+):?\s*</strong>\s*', block)
        if m4:
            sp = m4.group(1).strip()
            ln = _clean(block[m4.end():])
            if sp in SIX and len(ln) > 3:
                lines.append({'speaker': sp, 'line': ln})
            continue
        # 式2 (S02): 一个 p 块装整集, 按 <br> 分割逐行匹配行首 NAME:
        for seg in block.split('<br>'):
            m2 = re.match(r'\s*([A-Z][A-Z ]+):\s*', seg.strip())
            if not m2:
                continue  # 舞台说明/空行
            sp = m2.group(1).strip().title()  # RACHEL -> Rachel
            ln = _clean(seg[m2.end():])
            if sp in SIX and len(ln) > 3:
                lines.append({'speaker': sp, 'line': ln})
    return lines

def main():
    out = {'seasons': SEASONS, 'source': 'fangj.github.io/friends', 'data': {}}
    for season in SEASONS:
        all_lines, per_ep = [], {}
        for e in range(1, 25):
            if season == 2 and e == 13:
                per_ep['E13'] = 'merged-into-E12'
                continue
            # S02 E12/E13 是 Super Bowl 合并页
            fname = (f'{season:02d}12-0213.html'
                     if season == 2 and e == 12
                     else f'{season:02d}{e:02d}.html')
            try:
                ls = parse_ep(fetch(f'{BASE}/{fname}'))
                per_ep[f'E{e:02d}'] = len(ls)
                all_lines.extend(ls)
            except Exception as ex:
                per_ep[f'E{e:02d}'] = f'FAIL:{ex}'
            time.sleep(0.2)
        out['data'][f'S{season:02d}'] = {
            'per_episode_counts': per_ep, 'lines': all_lines}
        c = Counter(l['speaker'] for l in all_lines)
        print(f"S{season:02d}: {len(all_lines)} 句 · {c.most_common(3)}", flush=True)
    fname = 'friends_s' + ''.join(f'{s}' for s in SEASONS) + '_raw.json'
    json.dump(out, open(fname, 'w'), ensure_ascii=False, indent=1)
    print(f"已存 {fname}")

if __name__ == '__main__':
    main()
