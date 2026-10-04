# -*- coding: utf-8 -*-
"""
Bechberger 概念空间 2022 停点 — 四线并置数据与图
================================================================
四线（全部可复现，来源见下）:
  commits/year : GitHub API  lbechberger/ConceptualSpaces (133 commits)
  blog/year    : lucas-bechberger.de  WP REST wp/v2/posts (65 篇)
  pubs/year    : lucas-bechberger.de/publications/ 逐年条目
  cites/year   : OpenAlex author A5050395348 (counts_by_year, 合计 59)
对照线 field/year: OpenAlex 关键词 "conceptual spaces Gärdenfors" 年产量

产出: fig_bechberger_timeline.png + timeline_data.json
运行: /path/to/venv132/bin/python benches/FSSSS/bechberger_timeline.py
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEARS = list(range(2012, 2027))

COMMITS = {2017: 71, 2018: 25, 2019: 8, 2020: 23, 2021: 2, 2022: 4}
BLOG    = {2017: 23, 2018: 15, 2019: 5, 2020: 11, 2021: 10, 2022: 1}
PUBS    = {2012: 1, 2014: 2, 2016: 3, 2017: 12, 2018: 6, 2019: 3, 2020: 3, 2021: 10, 2023: 2}
CITES   = {2017: 4, 2018: 8, 2019: 8, 2020: 3, 2021: 9, 2022: 13, 2023: 5, 2024: 4, 2025: 4, 2026: 1}
FIELD   = {2016: 88, 2017: 86, 2018: 97, 2019: 111, 2020: 139, 2021: 145, 2022: 160, 2023: 178, 2024: 190, 2025: 176}


def series(d):
    return [d.get(y, 0) for y in YEARS]


def main():
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    panels = [
        (COMMITS, "GitHub commits  (repo activity)", "#1f4e79"),
        (BLOG,    "Blog posts  (site activity)", "#2e7d32"),
        (PUBS,    "Publications  (list entries)", "#8e24aa"),
        (CITES,   "Citations received  (OpenAlex)", "#c62828"),
    ]
    for ax, (d, title, color) in zip(axes, panels):
        ax.bar(YEARS, series(d), color=color, width=0.62)
        ax.set_ylabel(title, fontsize=9)
        for y, v in d.items():
            ax.text(y, v + 0.4, str(v), ha="center", va="bottom", fontsize=7)
        ax.axvline(2022, color="black", ls="--", lw=0.9)
        ax.margins(y=0.22)
        ax.grid(axis="y", alpha=0.25)
    axes[0].annotate("2022-01-20  last research commit (v1.3.2)",
                     xy=(2022, 4), xytext=(2019.3, 40), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].annotate("2022-09-29  last blog post", xy=(2022, 1),
                     xytext=(2019.4, 18), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[3].annotate("citations peak 2022", xy=(2022, 13), xytext=(2019.2, 12),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    fig.suptitle("Bechberger / ConceptualSpaces — output stops in 2022", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(os.path.dirname(__file__), "fig_bechberger_timeline.png")
    fig.savefig(out, dpi=140)
    print("fig →", out)

    data = {"commits": COMMITS, "blog": BLOG, "pubs": PUBS, "cites": CITES,
            "field_keyword": FIELD,
            "note": "commits=GitHub API; blog=WP REST; pubs=site list; cites=OpenAlex A5050395348; "
                    "field=OpenAlex keyword 'conceptual spaces Gärdenfors'"}
    json.dump(data, open(os.path.join(os.path.dirname(__file__), "timeline_data.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    # 领域对照单独小图
    fig2, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(list(FIELD), list(FIELD.values()), marker="o", color="#00695c")
    ax.axvline(2022, color="black", ls="--", lw=0.9)
    ax.set_title("Field keeps growing: OpenAlex works matching 'conceptual spaces Gärdenfors'", fontsize=9)
    ax.grid(alpha=0.25)
    fig2.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), "fig_field_growth.png")
    fig2.savefig(out2, dpi=140)
    print("fig →", out2)


if __name__ == "__main__":
    main()
