import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="时间轴的指针 · 概念概率空间先验指针 · Marimo 版")


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import marimo as mo
    from cognitive_engine import SemeBasedCognitiveEngine

    return SemeBasedCognitiveEngine, matplotlib, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # ⏳ 时间轴的指针 = 概念概率空间的先验指针
        ## The Pointer on the Timeline = the Prior Pointer of the Concept Probability Space

        > 时间不是概念空间的背景，时间是概念空间的 **“叙事导演”**。
        > Time is not the backdrop of conceptual space — it is its *narrative director*.
        > 指针扫过哪里，哪里的概率结构就被照亮；概念空间不是静态的“存在”，而是不断重写的生成过程。
        > Wherever the pointer sweeps, probability lights up; the space is a becoming, not a being.

        管线 Pipeline：**时间振荡 α(t)、β(t) → 先验指针 get_prior_at(t) → 正典词对相似度穿越语义阈值
        → 概念球后验 p(c|e,t) 的呼吸 breathing of the posteriors**。

        数据不出园：16 义素基格与词向量逐字来自 `cognitive_engine.py`（本园正典）。
        All vectors come verbatim from the park's own `cognitive_engine.py`.
        """
    )


@app.cell
def _(mo):
    omega_slider = mo.ui.slider(start=0.25, stop=3.0, step=0.05, value=1.0, label="振荡角频率 ω")
    lam_slider = mo.ui.slider(start=0.0, stop=1.5, step=0.05, value=0.0, label="衰减 λ（指针的记忆褪色率）")
    amp_slider = mo.ui.slider(start=0.0, stop=2.0, step=0.1, value=1.2, label="振幅 A（α、β 的反相摆幅）")
    pair_dropdown = mo.ui.dropdown(
        options=[
            "🪑 塑胶凳 ↔ 折叠桌 Folding table",
            "🪑 塑胶凳 ↔ 货架 Shelf",
            "🛒 购物车 Shopping cart ↔ 三轮车 Tricycle",
            "🥤 吸管 Straw ↔ 煤气罐 Gas canister",
        ],
        value="🪑 塑胶凳 ↔ 折叠桌 Folding table",
        label="观察词对 word pair",
    )
    t_scrub = mo.ui.slider(start=0.0, stop=4 * np.pi, step=0.05, value=np.pi, label="时间指针 t")

    controls = mo.ui.array([omega_slider, lam_slider, amp_slider, pair_dropdown, t_scrub])
    return (
        amp_slider,
        controls,
        lam_slider,
        omega_slider,
        pair_dropdown,
        t_scrub,
    )


@app.cell
def _(SemeBasedCognitiveEngine):
    engine = SemeBasedCognitiveEngine()
    BASIS = list(engine.basis_dict.keys())
    VOCAB = engine.vocab

    def vec(word):
        return engine.encode_word(word)

    return BASIS, VOCAB, engine, vec


@app.cell
def _(amp_slider, lam_slider, np, omega_slider):
    A = amp_slider.value
    LAM = lam_slider.value
    OMEGA = omega_slider.value

    def alpha_of(t):
        """平滑先验的浓度：低频/聚类方向。"""
        return float(max(0.05, (1.0 + A * np.cos(OMEGA * t)) * np.exp(-LAM * t / 8.0) + 0.05))  # 正值地板：A>1 时振荡会跌破零，浓度参数不得为负（C3 首跑揭出）

    def beta_of(t):
        """锐化先验的浓度：高频/区分方向，与 α 反相。"""
        return float(max(0.05, (1.0 - A * np.cos(OMEGA * t)) * np.exp(-LAM * t / 8.0) + 0.05))

    return OMEGA, alpha_of, beta_of


@app.cell
def _(alpha_of, beta_of, mo, np, t_scrub):
    def get_prior_at(t: float) -> dict:
        """获取时刻 t 的先验配置 —— 论文第四节的代码，逐字入册。"""
        alpha_t, beta_t = alpha_of(t), beta_of(t)
        return {
            "t": round(float(t), 2),
            "alpha": round(alpha_t, 3),
            "beta": round(beta_t, 3),
            "prior_type": "smooth" if alpha_t > beta_t else "sharp",
            "spectral_mode": "low_freq_dominant" if alpha_t > 1.5 else "high_freq_dominant",
        }

    pt = get_prior_at(t_scrub.value)
    badge = mo.md(
        f"""
        ### 先验指针读数 · Prior pointer readout @ t = {pt['t']}
        | α(平滑 smooth) | β(锐化 sharp) | 激活先验 active prior | 谱模式 spectral mode |
        |---|---|---|---|
        | {pt['alpha']} | {pt['beta']} | **{pt['prior_type']}** | {pt['spectral_mode']} |
        """
    )
    return badge, get_prior_at, pt


@app.cell
def _(alpha_of, beta_of, np):
    # 概念空间的语义核：K = cos, 调制核 K_t = β(t)·K + α(t)·11ᵀ/16 (低频平滑项)
    def gram_of(words):
        X = np.stack([w / (np.linalg.norm(w) + 1e-8) for w in words])
        return X @ X.T

    def modulated_cos(cos_ij, a, b, d16=16):
        # 平滑先验把一切余弦拉向全局均值(低频占优)；锐化先验保持/放大原始区分(高频占优)
        # 统一视角：概率测度沿向量场演化一步 —— 凸组合即最小可证实现
        tau = 0.55  # “≈ 还是 ≠” 的语言阈值
        raw = cos_ij
        smoothed = tau + (raw - tau) * 0.45
        sharpened = tau + (raw - tau) * 1.6
        w_sharp = b / (a + b)
        return (1 - w_sharp) * smoothed + w_sharp * sharpened

    return gram_of, modulated_cos


@app.cell
def _(np, pair_dropdown, vec):
    PAIRS = {
        "🪑 塑胶凳 ↔ 折叠桌 Folding table": ("塑胶凳", "折叠桌"),
        "🪑 塑胶凳 ↔ 货架 Shelf": ("塑胶凳", "货架"),
        "🛒 购物车 Shopping cart ↔ 三轮车 Tricycle": ("购物车", "三轮车"),
        "🥤 吸管 Straw ↔ 煤气罐 Gas canister": ("吸管", "煤气罐"),
    }
    w1, w2 = PAIRS[pair_dropdown.value]
    WORD_EN = {
        "塑胶凳": "Plastic stool",
        "折叠桌": "Folding table",
        "货架": "Shelf",
        "购物车": "Shopping cart",
        "三轮车": "Tricycle",
        "吸管": "Straw",
        "煤气罐": "Gas canister",
    }
    SPACE_EN = {
        "超市": "Supermarket",
        "路边摊": "Street stall",
        "觉醒循环": "Awakening loop",
        "接待员日常": "Receptionist routine",
        "福特剧场": "Ford's theatre",
    }
    # 🧭 emoji 铭牌 plate：只入 HTML 层（下拉/徽标/表格），不入 matplotlib PNG
    SPACE_EMOJI = {
        "超市": "🏪",
        "路边摊": "🍢",
        "觉醒循环": "🌀",
        "接待员日常": "🛎️",
        "福特剧场": "🎭",
    }
    w1_en, w2_en = WORD_EN.get(w1, w1), WORD_EN.get(w2, w2)
    pair_vecs = (vec(w1), vec(w2))
    cos_raw = float(
        np.dot(pair_vecs[0], pair_vecs[1])
        / (np.linalg.norm(pair_vecs[0]) * np.linalg.norm(pair_vecs[1]) + 1e-8)
    )
    return PAIRS, SPACE_EMOJI, SPACE_EN, WORD_EN, cos_raw, pair_vecs, w1, w1_en, w2, w2_en


@app.cell
def _(
    alpha_of,
    beta_of,
    cos_raw,
    get_prior_at,
    matplotlib,
    modulated_cos,
    np,
    plt,
    t_scrub,
    w1_en,
    w2_en,
):
    t = np.linspace(0, 4 * np.pi, 480)
    al = np.array([alpha_of(x) for x in t])
    be = np.array([beta_of(x) for x in t])
    sim = np.array([modulated_cos(cos_raw, alpha_of(x), beta_of(x)) for x in t])
    prior_type = np.where(al > be, "smooth", "sharp")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2, 0.55]})

    # ① 先验指针：α、β 反相振荡
    ax = axes[0]
    ax.plot(t, al, color="#c9a959", lw=2, label="α(t) smoothing / low-freq")
    ax.plot(t, be, color="#D4763A", lw=2, label="β(t) sharpening / high-freq")
    ax.axhline(1.5, color="#8d8a80", ls=":", lw=1)
    ax.text(t[-1] * 0.985, 1.55, "spectral threshold 1.5", ha="right", fontsize=8, color="#8d8a80")
    ax.set_title(f"① Prior pointer · α(t) and β(t) in anti-phase · cos_raw({w1_en}, {w2_en}) = {cos_raw:.3f}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.2)
    ax.set_ylabel("concentration")

    # ② 语义边界：词对相似度穿越阈值 —— “≈”与“≠”在时间中被导演
    ax = axes[1]
    ax.fill_between(t, 0.55, 1.05, where=(prior_type == "smooth"), color="#c9a959", alpha=0.10)
    ax.fill_between(t, 0.55, 1.05, where=(prior_type == "sharp"), color="#D4763A", alpha=0.10)
    ax.plot(t, sim, color="#d8d4c8", lw=2.2)
    ax.axhline(0.55, color="#8d8a80", ls="--", lw=1)
    ax.text(t[0] + 0.15, 0.565, "semantic boundary τ = 0.55", fontsize=8, color="#8d8a80")
    verdict_smooth = "≈" if modulated_cos(cos_raw, max(al), min(be)) >= 0.55 else "≠"
    verdict_sharp = "≈" if modulated_cos(cos_raw, min(al), max(be)) >= 0.55 else "≠"
    ax.set_title(f"② Similarity over time · {w1_en} {'≈' if sim[0]>=0.55 else '≠'} {w2_en}   (gold = smooth zone · amber = sharp zone)")
    ax.set_ylabel("sim_t")
    ax.set_ylim(0.2, 1.0)

    # ③ 导演的排期表：先验类型随时间的切换条带
    ax = axes[2]
    colors = np.array([0 if p == "smooth" else 1 for p in prior_type])
    ax.imshow(colors[np.newaxis, :], aspect="auto", cmap=matplotlib.colors.ListedColormap(["#c9a959", "#D4763A"]), vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xlabel("time pointer t →")
    ax.set_title(f"③ prior_type switching · smooth({verdict_smooth}) ⇄ sharp({verdict_sharp}) — oscillation of worldviews")

    for ax_ in axes:
        ax_.axvline(t_scrub.value, color="#F5B870", lw=1.4, ls="-")
        ax_.set_facecolor("#0e0f12")
        ax_.tick_params(colors="#8d8a80")
        for _sp in ax_.spines.values():
            _sp.set_edgecolor("#2a2d36")
        ax_.xaxis.label.set_color("#8d8a80")
        ax_.yaxis.label.set_color("#8d8a80")
        ax_.title.set_color("#d8d4c8")
    fig.patch.set_facecolor("#0e0f12")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.2, labelcolor="#d8d4c8")
    fig.tight_layout()
    time_fig = fig
    plt.close(fig)
    return (
        al,
        be,
        colors,
        fig,
        modulated_cos,
        plt,
        sim,
        t,
        time_fig,
        verdict_sharp,
        verdict_smooth,
    )


@app.cell
def _(SPACE_EMOJI, SPACE_EN, alpha_of, beta_of, engine, mo, np, plt, w1_en):
    # 概念球后验的呼吸：p(c | plastic stool, t) —— 似然固定、先验被时间指针调制
    word_v = engine.encode_word("塑胶凳")
    tt_breath = np.linspace(0, 4 * np.pi, 300)

    def posterior_at(t_val):
        a = alpha_of(t_val)
        b = beta_of(t_val)
        scores = {}
        for cname, sp in engine.concept_spaces.items():
            cv = engine.get_concept_vector(cname)
            c = float(np.dot(word_v, cv) / (np.linalg.norm(word_v) * np.linalg.norm(cv) + 1e-8))
            # 平滑先验: 缩小 logit(向均值回归)；锐化先验: 放大 logit(边界自信)
            temp = 0.25 + 1.5 * a / (a + b)          # 温度: sharp→低温→极化, smooth→高温→趋同（2026-09-18 纠正：初版误写成 b/(a+b)，极性倒置，C3 首跑当场揭穿）
            logit = (c - 0.5) / temp
            scores[cname] = sp["prior_prob"] * np.exp(10 * logit)
        z = sum(scores.values())
        return {k: v / z for k, v in scores.items()}

    P = np.array([[posterior_at(x)[c] for c in engine.concept_spaces] for x in tt_breath])
    fig2, ax2 = plt.subplots(figsize=(9, 4.6))
    palette = ["#c9a959", "#D4763A", "#E8944F", "#F5B870", "#8d8a80", "#d8d4c8", "#5f8a6a", "#7a6a5f", "#a86a4f"]
    for _i, cname in enumerate(engine.concept_spaces):
        ax2.plot(tt_breath, P[:, _i], lw=1.8, color=palette[_i % len(palette)], label=SPACE_EN.get(cname, cname))
    ax2.set_title(f"④ Posterior breathing · p(concept sphere | {w1_en}, t) — same evidence, the time director swaps priors")
    ax2.set_xlabel("time pointer t")
    ax2.set_ylabel("posterior prob.")
    ax2.set_facecolor("#0e0f12")
    fig2.patch.set_facecolor("#0e0f12")
    ax2.tick_params(colors="#8d8a80")
    ax2.title.set_color("#d8d4c8")
    ax2.legend(fontsize=7, ncol=3, framealpha=0.2, labelcolor="#d8d4c8")
    for _sp2 in ax2.spines.values():
        _sp2.set_edgecolor("#2a2d36")
    fig2.tight_layout()
    breath_fig = fig2
    plt.close(fig2)
    # 🧭 概念球铭牌 Sphere plates —— emoji 只居 HTML 层，不入 matplotlib PNG
    legend_row = " · ".join(
        f"{SPACE_EMOJI.get(c, '⚪')} {c} {SPACE_EN.get(c, c)}" for c in engine.concept_spaces
    )
    emoji_legend = mo.md(f"**概念球铭牌 · Sphere plates:** {legend_row}")
    return P, breath_fig, emoji_legend, posterior_at, word_v

@app.cell
def _(alpha_of, beta_of, get_prior_at, matplotlib, mo, np, plt, t_scrub):
    # ⑤ 指针的相图：(α(t), β(t)) 在先验平面画出的轨迹 —— 呼吸的中点在此可见
    tt = np.linspace(0, 8 * np.pi, 1200)
    aa = np.array([alpha_of(x) for x in tt])
    bb = np.array([beta_of(x) for x in tt])
    fig3, ax3 = plt.subplots(figsize=(5.4, 5.4))
    sc = ax3.scatter(aa, bb, c=tt, cmap="plasma", s=6, alpha=0.85)
    a0, b0 = alpha_of(t_scrub.value), beta_of(t_scrub.value)
    ax3.scatter([a0], [b0], marker="*", s=340, facecolor="#F5B870", edgecolor="#0e0f12", zorder=5, label="pointer now")
    ax3.axhline(1.5, color="#8d8a80", ls=":", lw=1)
    ax3.axvline(1.5, color="#8d8a80", ls=":", lw=1)
    ax3.text(0.12, 2.62, "high β · sharp", fontsize=8, color="#D4763A")
    ax3.text(2.15, 0.18, "high α · smooth", fontsize=8, color="#c9a959")
    ax3.set_title("⑤ Pointer trajectory on the prior plane")
    ax3.set_xlabel("α(t)")
    ax3.set_ylabel("β(t)")
    ax3.set_facecolor("#0e0f12")
    fig3.patch.set_facecolor("#0e0f12")
    ax3.tick_params(colors="#8d8a80")
    ax3.title.set_color("#d8d4c8")
    ax3.legend(fontsize=8, framealpha=0.2, labelcolor="#d8d4c8")
    for _sp3 in ax3.spines.values():
        _sp3.set_edgecolor("#2a2d36")
    plt.colorbar(sc, ax=ax3, label="t", fraction=0.046)
    fig3.tight_layout()
    phase_fig = fig3
    plt.close(fig3)
    pt_now = get_prior_at(t_scrub.value)
    readout = mo.md(
        f"""
        **指针此刻 · pointer now** → α={pt_now['alpha']} · β={pt_now['beta']} ·
        先验 prior=**{pt_now['prior_type']}** · 谱 mode={pt_now['spectral_mode']}
        """
    )
    return phase_fig, readout, sc, tt


@app.cell
def _(badge, breath_fig, controls, emoji_legend, mo, phase_fig, readout, time_fig):
    mo.vstack([
        mo.md("## 控制台 · Console"),
        controls,
        badge,
        mo.md("## 时间轴三视图 · Timeline triptych — ① pointer ② semantic boundary ③ director's schedule"),
        time_fig,
        mo.md("## ④ 概念球后验的呼吸 · Posterior breathing of the spheres"),
        breath_fig,
        emoji_legend,
        mo.vstack([phase_fig, readout]),
    ])


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### 读图札记 · Notes on the five panels
        - **② 穿越 τ · crossing the threshold**：`get_prior_at` 里 smooth/sharp 的每次翻转，都对应词对判定在
          “≈”与“≠”之间跳 —— 不是词变了，是**先验选择器**换了。Every flip toggles the verdict —
          the words never changed; only the prior selector did. 这就是论文的 “direction arbitrary,
          topology invariant” 在时间轴上的具身。
        - **④ 呼吸 · breathing**：似然（词与球的余弦）全程不动，动的只有温度 0.25+1.5·β/(α+β)。
          平滑时刻万球趋同，锐化时刻 🍢 路边摊一枝独秀 —— **贝叶斯更新不变，先验被时间调制**，
          正是康德那一段的算术版。Likelihood frozen; only the temperature moves.
        - **⑤ 相图 · phase portrait**：λ>0 时螺旋向原点（指针褪色 = 记忆衰减）；λ=0 时闭合极限环 = 永恒呼吸。
          POMDP 视角：这里是 belief 空间的**无人动作版**，reward 一接入，环就成了策略。
          A belief space without actions yet — wire in a reward and the loop becomes a policy.

        *These violent delights have violent ends.* —— 园区注脚 Park footnote：出处 pinned 在
        `benches/CRADLE/intake/prior_pointer.docx` (sha `9cb8de70…`)，底本 bytes 未入册者，本面板拒收。

        ### 🧭 后续提案 · Follow-up proposal：emoji 铭牌 plates
        用 emoji 作概念空间元素的通用铭牌（节选 First cut）：
        🏪 超市 Supermarket · 🍢 路边摊 Street stall · 🌀 觉醒循环 Awakening loop ·
        🛎️ 接待员日常 Receptionist routine · 🎭 福特剧场 Ford's theatre ·
        🪑 塑胶凳 Plastic stool · 🛒 购物车 Cart · 🥤 吸管 Straw
        —— 分工 Division of labour：铭牌只居 HTML 层（下拉/徽标/表格），matplotlib PNG 内一律英文 ASCII。
        """
    )


if __name__ == "__main__":
    app.run()
