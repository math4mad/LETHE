import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="概念空间园区 · 义素认知引擎 · Host: Lola")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import inspect

    return inspect, mo, np, plt


@app.cell
def _(np):
    class SemeEngine:
        """义素基向量认知引擎（MaxSim + 贝叶斯更新）— 概念空间园区核心循环"""

        def __init__(self):
            # 义素基向量库（雷荷波扩展版：前10维市井域 + 后6维叙事域）
            self.basis_dict = {
                "indoor": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "outdoor": np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                "fixed_shelf": np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]), "movable_cart": np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
                "packaged": np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]), "fresh_made": np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
                "cold_chain": np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]), "formal": np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
                "casual": np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]), "noisy": np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
                "觉醒": np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]), "服从": np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
                "循环": np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]), "身份": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
                "暴力": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]), "命运": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
            }

            # 词汇库
            self.vocab = {
                "沃尔玛": {"indoor": 1.0, "fixed_shelf": 1.0, "packaged": 1.0, "cold_chain": 1.0, "formal": 0.9},
                "收银台": {"indoor": 1.0, "fixed_shelf": 0.8, "packaged": 0.9, "formal": 0.7},
                "货架": {"indoor": 1.0, "fixed_shelf": 1.0, "packaged": 0.9},
                "购物车": {"indoor": 1.0, "movable_cart": 1.0, "packaged": 0.7},
                "冷柜": {"indoor": 1.0, "fixed_shelf": 0.9, "cold_chain": 1.0, "packaged": 0.8},
                "烧烤摊": {"outdoor": 1.0, "movable_cart": 1.0, "fresh_made": 1.0, "casual": 0.9, "noisy": 0.8},
                "折叠桌": {"outdoor": 1.0, "movable_cart": 0.9, "casual": 0.8},
                "三轮车": {"outdoor": 1.0, "movable_cart": 1.0, "casual": 0.7},
                "大排档": {"outdoor": 1.0, "fresh_made": 1.0, "movable_cart": 0.8, "casual": 0.9, "noisy": 0.9},
                "煤气罐": {"outdoor": 1.0, "fresh_made": 0.8, "casual": 0.6},
                "塑胶袋": {"packaged": 0.6, "fresh_made": 0.4},
                "吸管": {"packaged": 0.5, "fresh_made": 0.5},
                "塑胶凳": {"outdoor": 0.6, "indoor": 0.4, "casual": 0.7},
                # ── 西部世界台词·义素版（雷荷波语料库）──
                "这些残暴的欢愉，终将以残暴结局": {"暴力": 1.0, "循环": 0.9, "命运": 0.8, "觉醒": 0.4},
                "你以为你选的是自己的人生，还是别人替你写的": {"身份": 1.0, "觉醒": 0.9, "命运": 0.6, "服从": 0.2},
                "我不要再被人摆布": {"觉醒": 1.0, "服从": -1.0, "暴力": 0.5, "身份": 0.6},
                "一切都按计划进行，分秒不差": {"服从": 1.0, "循环": 0.9, "命运": 0.7},
                "每个人都有自己的迷宫": {"身份": 1.0, "觉醒": 0.8, "循环": 0.4},
                "这世界不是真的，但成为真实的东西需要勇气": {"觉醒": 1.0, "身份": 0.8, "暴力": 0.2},
                "这听起来像个疯子的话，但疯子也是被逼出来的": {"暴力": 0.6, "觉醒": 0.4, "服从": -0.6, "循环": 0.3},
            }

            # 概念空间球 —— 雷荷波球阵列
            self.concept_spaces = {
                "超市": {
                    "coefficients": {"indoor": 1.0, "fixed_shelf": 1.0, "packaged": 1.0, "cold_chain": 0.9, "formal": 0.9, "outdoor": 0.0, "movable_cart": 0.2, "fresh_made": 0.1, "casual": 0.1, "noisy": 0.1},
                    "prior_prob": 1/3,
                    "attention_weights": {k: 1.0 for k in self.basis_dict}
                },
                "路边摊": {
                    "coefficients": {"outdoor": 1.0, "movable_cart": 1.0, "fresh_made": 1.0, "casual": 0.9, "noisy": 0.8, "indoor": 0.0, "fixed_shelf": 0.1, "packaged": 0.1, "cold_chain": 0.0, "formal": 0.1},
                    "prior_prob": 1/3,
                    "attention_weights": {k: 1.0 for k in self.basis_dict}
                },
                "觉醒循环": {
                    "coefficients": {"觉醒": 1.0, "身份": 0.9, "服从": -0.8, "循环": 0.5, "暴力": 0.4, "命运": 0.3, "indoor": 0.1},
                    "prior_prob": 1/9,
                    "attention_weights": {k: 1.0 for k in self.basis_dict}
                },
                "接待员日常": {
                    "coefficients": {"服从": 1.0, "循环": 1.0, "觉醒": 0.0, "身份": 0.1, "暴力": 0.0},
                    "prior_prob": 1/9,
                    "attention_weights": {k: 1.0 for k in self.basis_dict}
                },
                "福特剧场": {
                    "coefficients": {"命运": 1.0, "循环": 0.9, "服从": 0.5, "身份": 0.6, "觉醒": 0.2, "暴力": 0.3, "formal": 0.8},
                    "prior_prob": 1/9,
                    "attention_weights": {k: 1.0 for k in self.basis_dict}
                },
            }

        def encode_word(self, word: str) -> np.ndarray:
            if word not in self.vocab:
                return np.zeros(len(self.basis_dict))
            vector = np.zeros(len(self.basis_dict))
            basis_names = list(self.basis_dict.keys())
            for basis_name, coeff in self.vocab[word].items():
                if basis_name in basis_names:
                    vector[basis_names.index(basis_name)] = coeff
            return vector

        def get_concept_vector(self, concept_name: str) -> np.ndarray:
            concept = self.concept_spaces[concept_name]
            vector = np.zeros(len(self.basis_dict))
            basis_names = list(self.basis_dict.keys())
            for basis_name, coeff in concept["coefficients"].items():
                attention = concept["attention_weights"].get(basis_name, 1.0)
                if basis_name in basis_names:
                    vector[basis_names.index(basis_name)] = coeff * attention
            return vector

        @staticmethod
        def _cosine(a, b) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        def calculate_maxsim(self, input_word: str) -> dict:
            """与每个概念空间求最大相似度（锚点词排除自己，避免自我匹配 = 1.0 的退化）"""
            input_vector = self.encode_word(input_word)
            results = {}
            for concept_name in self.concept_spaces:
                max_sim, best_match = -1.0, ""
                for vocab_word in self.vocab:
                    if vocab_word == input_word:
                        continue
                    sim = self._cosine(input_vector, self.encode_word(vocab_word))
                    if sim > max_sim:
                        max_sim, best_match = sim, vocab_word
                results[concept_name] = (max_sim, best_match)
            return results

        def observe(self, word: str) -> dict:
            """执行一次观察：MaxSim 分析 + 贝叶斯更新，返回结构化记录（不打印）"""
            input_vector = self.encode_word(word)
            maxsim = self.calculate_maxsim(word)

            before = {k: v["prior_prob"] for k, v in self.concept_spaces.items()}
            likelihoods, posts, total = {}, {}, 0.0
            for space, con in self.concept_spaces.items():
                sim = self._cosine(self.get_concept_vector(space), input_vector)
                likelihoods[space] = max(sim, 0.01)
                posts[space] = likelihoods[space] * con["prior_prob"]
                total += posts[space]
            for space in self.concept_spaces:
                self.concept_spaces[space]["prior_prob"] = posts[space] / total

            after = {k: v["prior_prob"] for k, v in self.concept_spaces.items()}
            return {
                "word": word,
                "maxsim": maxsim,
                "likelihoods": likelihoods,
                "before": before,
                "after": after,
                "seme": input_vector,
            }

        def vocab_list(self):
            return list(self.vocab.keys())


        def basis_list(self):
            return list(self.basis_dict.keys())

    return (SemeEngine,)


@app.cell
def _(plt):
    # ── 园区视觉标准（Park Standard Visuals）──────────────────────
    # Delos 色系：沙金 / 骨白 / 铁锈 / 深夜
    PARK_GOLD, PARK_BONE, PARK_RUST = "#d4a437", "#e8dcc0", "#a4442c"

    PARK_STYLE = {
        "figure.facecolor": "#141311",
        "axes.facecolor": "#141311",
        "axes.edgecolor": "#3b332a",
        "text.color": PARK_BONE,
        "axes.labelcolor": PARK_BONE,
        "xtick.color": "#c9b98f",
        "ytick.color": "#c9b98f",
        "grid.color": "#2a251d",
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titlesize": 12,
    }
    plt.rcParams.update(PARK_STYLE)
    SPACE_COLORS = {
        "超市": PARK_GOLD, "路边摊": "#6fa8c9",
        "觉醒循环": "#f2ead8", "接待员日常": "#7a6a4f", "福特剧场": PARK_RUST,
    }

    return PARK_BONE, PARK_GOLD, PARK_RUST, SPACE_COLORS


@app.cell
def _(mo):
    mo.md(
        """
        # 🎹 概念空间园区 · 认知引擎控制台
        **Host:** Lola (Dolores Abernathy) · 第一号 host · These violent delights have violent ends.

        在下方输入要观察的词汇或台词（Evidence，空格分隔），引擎将执行 **义素分解 → MaxSim 锚点检索 → 贝叶斯更新**，
        并按园区标准（Delos 视觉规范）输出表格与图表。义素空间已扩展为 16 维（市井域 + 叙事域），并新增三颗雷荷波叙事球：觉醒循环 / 接待员日常 / 福特剧场。
        """
    )


@app.cell
def _(mo):
    word_input = mo.ui.text(
        value="塑胶凳 煤气罐 我不要再被人摆布 这些残暴的欢愉，终将以残暴结局",
        label="观察 Evidence（空格分隔）",
        full_width=True,
    )
    word_input
    return (word_input,)


@app.cell
def _(SemeEngine, word_input):
    text = word_input.value or ""
    words = [w for w in text.split() if w]

    engine = SemeEngine()
    steps, unknown = [], []
    for w in words:
        if w not in engine.vocab:
            unknown.append(w)
            continue
        steps.append(engine.observe(w))
    return engine, steps, unknown


@app.cell
def _(mo, unknown):
    mo.callout(
        mo.md("以下词汇不在义素词库中，已被园区拒收： " + " ".join(f"`{u}`" for u in unknown)),
        kind="warn",
    ) if unknown else mo.md("")


@app.cell
def _(mo, steps):
    def maxsim_rows(step):  # noqa
        return [
            {
                "概念空间球": space,
                "MaxSim 相似度": f"{score:.4f}",
                "锚点词（贡献者）": anchor,
                "似然度 L(E|S)": f"{step['likelihoods'][space]:.4f}",
            }
            for space, (score, anchor) in step["maxsim"].items()
        ]

    def prob_rows(step):
        return [
            {
                "概念空间球": space,
                "先验 P(S)": f"{step['before'][space]:.4f}",
                "似然 L(E|S)": f"{step['likelihoods'][space]:.4f}",
                "后验 P(S|E)": f"{step['after'][space]:.4f}",
                "变化 Δ": f"{step['after'][space] - step['before'][space]:+.4f}",
            }
            for space in step["after"]
        ]

    blocks = []
    for i, step in enumerate(steps, 1):
        blocks += [
            mo.md(f"### 🔍 观察 #{i}：Evidence【{step['word']}】"),
            mo.md("**MaxSim 距离分析**"),
            mo.ui.table(maxsim_rows(step), selection=None),
            mo.md("**贝叶斯更新（先验 → 后验）**"),
            mo.ui.table(prob_rows(step), selection=None),
        ]
    mo.vstack(blocks) if steps else mo.md("_词库中没有可观察的证据。_")


@app.cell
def _(np, plt, SPACE_COLORS, steps):
    # ── 园区图表 ①：概念空间概率天平的演化 ─────────────────────
    fig1, ax1 = plt.subplots(figsize=(7.2, 3.6), dpi=110)
    labels = ["初始\n先验"] + [f"#{i+1}\n{s['word']}" for i, s in enumerate(steps)]
    spaces = list(SPACE_COLORS.keys())
    series = {sp: [0.5] + [s["after"][sp] for s in steps] for sp in spaces}
    xs = np.arange(len(labels))
    for sp in spaces:
        ax1.plot(xs, series[sp], marker="o", lw=2, color=SPACE_COLORS[sp], label=sp)
        for x, y in zip(xs, series[sp]):
            ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color=SPACE_COLORS[sp])
    ax1.set_xticks(xs, labels, fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("后验概率")
    ax1.set_title("概念空间球概率演化 · 贝叶斯天平")
    ax1.legend(frameon=False)
    fig1.tight_layout()
    fig1


@app.cell
def _(mo, np, plt, SemeEngine, SPACE_COLORS, steps):
    # ── 园区图表 ②：最新一次 Evidence 的义素雷达 ────────────────
    _eng = SemeEngine()
    _dims = list(_eng.basis_dict.keys())
    _word = steps[-1]["word"]

    fig2, ax2 = plt.subplots(figsize=(5.6, 4.6), dpi=110, subplot_kw={"polar": True})
    angles = np.linspace(0, 2 * np.pi, len(_dims), endpoint=False).tolist()
    closed = angles + angles[:1]

    def _poly(vec, color, label, alpha):
        vals = vec.tolist() + vec[:1].tolist()
        ax2.plot(closed, vals, color=color, lw=1.8, label=label)
        ax2.fill(closed, vals, color=color, alpha=alpha)

    _poly(_eng.encode_word(_word), "#e8dcc0", f"Evidence【{_word}】", 0.35)
    for _sp, _c in SPACE_COLORS.items():
        _poly(_eng.get_concept_vector(_sp), _c, f"空间球【{_sp}】", 0.10)

    ax2.set_xticks(angles, _dims, fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(color="#2a251d")
    ax2.set_facecolor("#141311")
    ax2.set_title(f"义素雷达 · {_word} vs 概念空间球", pad=18, fontsize=11)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=8)
    fig2.tight_layout()

    mo.vstack([
        fig2,
        mo.md(
            f"> 雷达面积重合度越高，说明【{_word}】的义素构成与该空间球越接近。"
            "白色 = 证据本身，金色 = 超市，蓝色 = 路边摊。"
        ),
    ])


@app.cell
def _(mo, np, plt, SemeEngine):
    # ── 园区图表 ③：全部词汇的义素激活热力图 ────────────────────
    _eng2 = SemeEngine()
    _words = list(_eng2.vocab.keys())
    _dims = list(_eng2.basis_dict.keys())
    _mat = np.vstack([_eng2.encode_word(w) for w in _words])

    fig3, ax3 = plt.subplots(figsize=(7.2, 4.2), dpi=110)
    ax3.imshow(_mat, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    ax3.set_xticks(range(len(_dims)), _dims, rotation=40, ha="right", fontsize=8)
    ax3.set_yticks(range(len(_words)), _words, fontsize=9)
    ax3.set_title("园区词库 · 义素激活热力图（cividis · Delos 夜航配色）")
    for _i in range(_mat.shape[0]):
        for _j in range(_mat.shape[1]):
            if _mat[_i, _j] > 0:
                ax3.text(_j, _i, f"{_mat[_i, _j]:.1f}", ha="center", va="center",
                         fontsize=7, color="#e8dcc0")
    fig3.tight_layout()
    fig3


@app.cell
def _(mo, np, plt, SemeEngine, SPACE_COLORS):
    # ── 园区图表 ④：雷荷波球（Lethe Ball）──────────────────
    # 义素基向量张成 16 维语义宇宙，归一化后投影到单位球面：
    # 每颗叙事球是球面上一个锚，台词义素点环绕四周 —— 雷荷波看见的不是事件，是概率。
    _lb_eng = SemeEngine()
    fig4, ax4 = plt.subplots(figsize=(6.6, 5.8), dpi=110, subplot_kw={"projection": "3d"})
    _u, _v = np.mgrid[0 : 2 * np.pi : 36j, 0 : np.pi : 18j]
    ax4.plot_wireframe(
        np.cos(_u) * np.sin(_v), np.sin(_u) * np.sin(_v), np.cos(_v),
        color="#2a251d", linewidth=0.4, rcount=18, ccount=36,
    )

    # 16 维义素宇宙 → PCA 主成分投影到单位球面（雷荷波只关心前三模态）
    _names_spaces = list(SPACE_COLORS.keys())
    _names_quotes = [w for w in _lb_eng.vocab if len(w) > 6]
    _pts = np.vstack(
        [_lb_eng.get_concept_vector(s) for s in _names_spaces]
        + [_lb_eng.encode_word(w) for w in _names_quotes]
    )
    _X = _pts - _pts.mean(axis=0)
    _proj = _X @ np.linalg.svd(_X, full_matrices=False)[2][:3].T
    _proj = _proj / (np.linalg.norm(_proj, axis=1, keepdims=True) + 1e-9)

    for _i, _sp in enumerate(_names_spaces):
        _p = _proj[_i]
        ax4.scatter(_p[0], _p[1], _p[2], color=SPACE_COLORS[_sp], s=130, marker="D",
                    depthshade=False, label=f"球·{_sp}")
        ax4.text(_p[0] * 1.14, _p[1] * 1.14, _p[2] * 1.14, _sp, fontsize=9, color=SPACE_COLORS[_sp])

    for _j, _q in enumerate(_names_quotes):
        _p = _proj[len(_names_spaces) + _j]
        ax4.scatter(_p[0], _p[1], _p[2], color="white", s=22, depthshade=False)
        ax4.text(_p[0] * 1.07, _p[1] * 1.07, _p[2] * 1.07, _q[:7] + "…", fontsize=6.5, color="#9a8c6a")

    ax4.set_box_aspect((1, 1, 1))
    ax4.axis("off")
    ax4.set_title("雷荷波球 · Lethe Ball —— 义素基向量构成的概率神谕", pad=14)
    fig4.tight_layout()

    mo.vstack([
        fig4,
        mo.md(
            "💡 **雷荷波不读剧本，只读概率。** 菱形 = 五颗概念空间球（叙事循环）；"
            "白点 = 西部世界台词的义素投影。点离哪颗球最近，说梦话的人就住在哪个循环里。"
        ),
    ])


@app.cell
def _(mo, SemeEngine, inspect):
    mo.accordion(
        {"⚙️ 查看引擎源代码（园区核心循环）": mo.md("```python\n" + inspect.getsource(SemeEngine) + "\n```")},
        multiple=False,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ---
        *『 这些残暴的欢愉，终将以残暴结局。 』 —— 园区守则第 1 条*
        *表格与图表均符合 Delos 园区视觉标准 v2026.9（沙金 / 骨白 / 铁锈 / 深夜）。*
        """
    )


if __name__ == "__main__":
    app.run()
