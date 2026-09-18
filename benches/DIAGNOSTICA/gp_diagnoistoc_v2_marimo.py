import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="GP 漂移诊断器 v2 · 贝叶斯模型比较 + 谱截断 · Marimo 版")


@app.cell
def _():
    import warnings

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.exceptions import ConvergenceWarning
    from scipy.special import jacobi

    # length scales are deliberately PINNED to hypothesis values for model
    # comparison, so boundary-convergence notices are expected -> silence them
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    return (
        ConstantKernel,
        GaussianProcessRegressor,
        RBF,
        jacobi,
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # 🌊 GP 漂移诊断器 v2（贝叶斯模型比较 + 可证伪谱截断）

        管线：**三种 ground-truth regime（渐进/阶跃/折线）→ 双假设 GP 核（H_smooth vs H_abrupt）
        → 对数边际似然 · 贝叶斯因子 P(abrupt|D) → 后验轨迹 μ(t) 的 Legendre 谱展开
        → 按重构 RMSE 定量截断**。

        与 v1 的区别：检测不再靠 `mean+2σ` 拍脑袋阈值，截断不再是常数信号循环论证，
        且检测器在压力测试下**允许出错**（ramp 的拐点导数不连续，报 ABRUPT 可辩护）。
        """
    )


@app.cell
def _(mo):
    regime_dropdown = mo.ui.dropdown(
        options=["gradual", "step", "ramp"],
        value="gradual",
        label="聚焦 regime",
    )
    noise_slider = mo.ui.slider(start=0.01, stop=0.20, step=0.01, value=0.05, label="观测噪声 σ")
    ls_smooth_slider = mo.ui.slider(start=1.0, stop=8.0, step=0.5, value=3.0, label="H_smooth 长度尺度")
    ls_abrupt_slider = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=0.3, label="H_abrupt 长度尺度")
    order_slider = mo.ui.slider(start=3, stop=20, step=1, value=15, label="谱展开最高阶数")
    tol_slider = mo.ui.slider(start=0.005, stop=0.10, step=0.005, value=0.02, label="截断容差（× 信号σ）")

    controls = mo.vstack(
        [
            mo.hstack([regime_dropdown, noise_slider], justify="space-between"),
            mo.hstack([ls_smooth_slider, ls_abrupt_slider], justify="space-between"),
            mo.hstack([order_slider, tol_slider], justify="space-between"),
        ]
    )
    controls
    return (
        ls_abrupt_slider,
        ls_smooth_slider,
        noise_slider,
        order_slider,
        regime_dropdown,
        tol_slider,
    )


@app.cell
def _(
    ConstantKernel,
    GaussianProcessRegressor,
    RBF,
    ls_abrupt_slider,
    ls_smooth_slider,
    noise_slider,
    np,
):
    """① 数据模拟（三种 regime）→ ② 双假设 GP 拟合 → ③ 贝叶斯因子判定"""

    noise_std = noise_slider.value

    def sigmoid(x, center, width):
        return 1 / (1 + np.exp(-(x - center) / width))

    T = np.linspace(0, 10, 200)
    X = T.reshape(-1, 1)
    t_grid = np.linspace(0, 10, 500).reshape(-1, 1)

    truth = {
        "gradual": 1 - sigmoid(T, 5, 1.5),
        "step": 1 - sigmoid(T, 5, 0.05),
        "ramp": np.clip(1 - (T - 3) / 4, 0, 1),
    }
    # NOTE: ramp 在两处拐点导数不连续，报 ABRUPT 其实可辩护
    expected = {"gradual": "GRADUAL", "step": "ABRUPT", "ramp": "GRADUAL"}

    def fit_gp(y, length_scale):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale, (0.999 * length_scale, 1.001 * length_scale)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=noise_std**2, normalize_y=True, n_restarts_optimizer=5
        )
        gp.fit(X, y)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        return gp, lml

    bayes = {}
    for key, clean in truth.items():
        np.random.seed(42)
        y = clean + np.random.normal(0, noise_std, len(T))

        gp_s, lml_s = fit_gp(y, ls_smooth_slider.value)
        gp_a, lml_a = fit_gp(y, ls_abrupt_slider.value)
        mu_s, sd_s = gp_s.predict(t_grid, return_std=True)
        mu_a, sd_a = gp_a.predict(t_grid, return_std=True)

        # Bayes factor（等权先验 50/50）
        post_abrupt = 1 / (1 + np.exp(-(lml_a - lml_s)))
        verdict = "ABRUPT" if post_abrupt > 0.5 else "GRADUAL"
        mu_win = mu_a if post_abrupt > 0.5 else mu_s

        bayes[key] = dict(
            y=y,
            mu_s=mu_s,
            sd_s=sd_s,
            mu_a=mu_a,
            sd_a=sd_a,
            mu_win=mu_win,
            lml_s=lml_s,
            lml_a=lml_a,
            post_abrupt=post_abrupt,
            verdict=verdict,
            expected=expected[key],
            ok=verdict == expected[key],
        )

    return T, bayes, expected, fit_gp, noise_std, t_grid, truth


@app.cell
def _(bayes, jacobi, np, order_slider, t_grid, tol_slider):
    """④ 对后验轨迹 μ(t) 做真正的 Legendre（Jacobi α=β=0）谱展开 → ⑤ RMSE 定量截断"""

    max_order = int(order_slider.value)
    x_pts = np.linspace(-1, 1, 800)

    def legendre(x, n):
        return jacobi(n, 0.0, 0.0)(x)

    spectra = {}
    for bkey, brec in bayes.items():
        mu = brec["mu_win"]
        mu_on_x = np.interp(x_pts, (t_grid.flatten() - 5) / 5, mu)

        coeffs = np.zeros(max_order + 1)
        for n in range(max_order + 1):
            P = legendre(x_pts, n)
            coeffs[n] = np.trapezoid(mu_on_x * P, x_pts) / np.trapezoid(P**2, x_pts)

        # 累积重构 RMSE 曲线（按阶数顺序）
        recon = np.zeros_like(x_pts)
        rmse_curve = np.zeros(max_order + 1)
        for n in range(max_order + 1):
            recon += coeffs[n] * legendre(x_pts, n)
            rmse_curve[n] = np.sqrt(np.mean((recon - mu_on_x) ** 2))

        # 贪心剪枝：最少多少项（按 |c| 最大优先）使 RMSE < 容差
        tolerance = tol_slider.value * np.std(mu_on_x)
        order_idx = np.argsort(-np.abs(coeffs))
        n_needed = max_order + 1
        for k in range(1, max_order + 2):
            sel = np.sort(order_idx[:k])
            rec_k = sum(coeffs[j] * legendre(x_pts, j) for j in sel)
            if np.sqrt(np.mean((rec_k - mu_on_x) ** 2)) < tolerance:
                n_needed = k
                break

        spectra[bkey] = dict(
            coeffs=coeffs,
            rmse_curve=rmse_curve,
            n_needed=n_needed,
            tolerance=tolerance,
            top_orders=[int(j) for j in np.sort(order_idx[: min(n_needed, 6)])],
        )

    return legendre, max_order, spectra, x_pts


@app.cell
def _(T, bayes, max_order, np, plt, spectra, t_grid):
    """⑥ 3×3 可视化：后验对比 | Legendre 谱 | RMSE-截断曲线"""

    fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex="col")

    for row, (pkey, prec) in enumerate(bayes.items()):
        ax = axes[row, 0]
        ax.scatter(T, prec["y"], c="k", s=8, alpha=0.3, label="noisy data")
        ax.plot(t_grid, prec["mu_s"], "b-", lw=2, label=f"H_smooth  (LML={prec['lml_s']:.1f})")
        ax.fill_between(
            t_grid.flatten(),
            prec["mu_s"] - 2 * prec["sd_s"],
            prec["mu_s"] + 2 * prec["sd_s"],
            alpha=0.15,
            color="b",
        )
        ax.plot(t_grid, prec["mu_a"], "r-", lw=2, label=f"H_abrupt  (LML={prec['lml_a']:.1f})")
        ax.fill_between(
            t_grid.flatten(),
            prec["mu_a"] - 2 * prec["sd_a"],
            prec["mu_a"] + 2 * prec["sd_a"],
            alpha=0.15,
            color="r",
        )
        ax.set_ylabel("activity")
        mark = "OK" if prec["ok"] else "MISS"
        ax.set_title(
            f"truth: {pkey}  |  verdict: {prec['verdict']} ({mark}), "
            f"P(abrupt)={prec['post_abrupt']:.3f}"
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        rep = spectra[pkey]
        ax = axes[row, 1]
        ax.bar(range(max_order + 1), np.abs(rep["coeffs"]), color="steelblue", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("basis order n")
        ax.set_ylabel("|c_n|")
        ax.set_title("Legendre spectrum of posterior trajectory")
        ax.grid(alpha=0.3, axis="y")

        ax = axes[row, 2]
        ax.plot(range(max_order + 1), rep["rmse_curve"], "go-", label="cumulative RMSE")
        ax.axhline(rep["tolerance"], color="orange", ls="--", label="tolerance (x signal std)")
        ax.axvline(min(rep["n_needed"], max_order + 1) - 1, color="gray", ls=":",
                   label=f"kept={rep['n_needed']}")
        ax.set_yscale("log")
        ax.set_xlabel("number of terms kept")
        ax.set_ylabel("reconstruction RMSE")
        ax.set_title("Truncation cost is measurable (falsifiable)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.close(fig)
    fig
    return (fig,)


@app.cell
def _(bayes, mo, regime_dropdown, spectra):
    """⑦ 诊断结论面板"""

    focus_key = regime_dropdown.value or "gradual"
    fr = bayes[focus_key]
    frep = spectra[focus_key]
    n_correct = sum(r["ok"] for r in bayes.values())

    table = mo.ui.table(
        {
            "regime": list(bayes.keys()),
            "LML smooth": [round(float(r["lml_s"]), 2) for r in bayes.values()],
            "LML abrupt": [round(float(r["lml_a"]), 2) for r in bayes.values()],
            "P(abrupt|D)": [round(float(r["post_abrupt"]), 4) for r in bayes.values()],
            "verdict": [r["verdict"] for r in bayes.values()],
            "expected": [r["expected"] for r in bayes.values()],
            "result": ["✅" if r["ok"] else "❌" for r in bayes.values()],
        }
    )

    mo.md(
        f"""
        ### 📊 贝叶斯漂移判定（准确率 {n_correct}/{len(bayes)}）

        {table}

        ### 🔍 聚焦：`{focus_key}`

        - 判定：**{fr['verdict']}**，P(abrupt|D) = `{fr['post_abrupt']:.4f}`
          （log Bayes factor = `{fr['lml_a'] - fr['lml_s']:.2f}` nats）
        - 谱截断：保留 **{frep['n_needed']} / {len(frep['coeffs'])}** 项达到容差
          `{frep['tolerance']:.4f}`，选中阶：`{frep['top_orders']}`

        > 💡 ramp 在拐点处导数不连续，被判 ABRUPT 属于"可辩护的错误"；
        > step 的 Legendre 系数衰减极慢（Gibbs 现象），截断必须保留几乎全部项 —— 这些正是**真实**的可证伪行为。
        """
    )


if __name__ == "__main__":
    app.run()
