# -*- coding: utf-8 -*-
# 合成层: 透明渲染图 -> 米白底 + 双语标注 + 题头图例 (宋体兜底中文)
import json, sys
from PIL import Image, ImageDraw, ImageFont
RAW = "/tmp/topos_blender_raw.png"; OUT = sys.argv[1] if len(sys.argv) > 1 else "topos-blender-v2.png"
lab = json.load(open(RAW + ".labels.json"))
W, H = 1600, 1100
bg = Image.new("RGB", (W, H), (247, 242, 229))
fg = Image.open(RAW).convert("RGBA"); bg.paste(fg, (0, 0), fg)
d = ImageDraw.Draw(bg)
G = lambda n: ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf", n)
S = lambda n: ImageFont.truetype("/System/Library/Fonts/Supplemental/Songti.ttc", n)
F_LAT, F_LAT_S, F_CN, F_TTL, F_SUB = G(26), G(19), S(20), G(34), S(21)
INK = (47, 42, 36); RED = (140, 47, 47); GOLD = (176, 118, 42); TEAL = (46, 143, 163); GRAY = (150, 142, 125); MUT = (110, 100, 85)
def center(t, x, y, f, c):
    w = d.textlength(t, font=f); d.text((x - w/2, y), t, font=f, fill=c)
for k, (x, y, cn, latin, kind) in lab.items():
    c = RED if kind == "polity" else INK
    f1 = F_LAT if kind == "polity" else F_LAT_S
    center(latin.upper(), x, y - 46, f1, c)
    center(cn, x, y - 16, F_CN, MUT)
d.text((46, 28), "GARDEN TOPOGRAPHY", font=F_TTL, fill=INK)
d.text((46, 78), "园体地形志 · 等距拓扑 · 十九点二十二楼", font=F_SUB, fill=MUT)
d.text((46, 108), "ISO-TOPOLOGY · only relations are real", font=F_LAT_S, fill=(140, 130, 110))
ly = H - 64
for i, (name, cn, col, dash) in enumerate([("git road", "大路", GOLD, 1), ("sea route", "海路", TEAL, 1), ("in-repo lane", "仓内巷", GRAY, 0)]):
    x0 = 46 + i*430
    if dash: d.rectangle([x0, ly+8, x0+52, ly+14], fill=col)
    else:
        for j in range(4): d.rectangle([x0+j*14, ly+9, x0+j*14+7, ly+13], fill=col)
    d.text((x0+62, ly-2), name, font=F_LAT_S, fill=INK)
    d.text((x0+62+d.textlength(name, font=F_LAT_S)+8, ly-1), cn, font=F_CN, fill=MUT)
d.rectangle([46, H-104, 60, H-90], outline=RED, width=2)
d.text((68, H-106), "polity", font=F_LAT_S, fill=RED)
d.text((68 + d.textlength("polity", font=F_LAT_S) + 8, H-105), "域体", font=F_CN, fill=RED)
bg.save(OUT, quality=95); print("COMPOSITED", OUT)
