import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="GP 漂移诊断器 · Jacobi 基自适应管线 · Marimo 版")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from scipy.special import jacobi

    return ConstantKernel, GaussianProcessRegressor, RBF, jacobi, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # 🌊 GP 漂移诊断器（夜市 → 便利店 渐进漂移）

        管线：**数据模拟 → GP 拟合 → 后验导数诊断漂移强度 → 超参数 α(t), β(t) 自适应演化
        → Jacobi 基展开 → 系数幅值截断剪枝**。

        拖动下方滑块，整条管线会实时重算。
        """
    )


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(start=0.01, stop=0.20, step=0.01, value=0.05, label="观测噪声 σ")
    center_slider = mo.ui.slider(start=2.0, stop=8.0, step=0.1, value=5.0, label="转移中心")
    width_slider = mo.ui.slider(start=0.5, stop=4.0, step=0.1, value=1.5, label="转移宽度")
    length_slider = mo.ui.slider(start=0.5, stop=5.0, step=0.1, value=2.0, label="RBF length scale")
    order_slider = mo.ui.slider(start=1, stop=8, step=1, value=5, label="Jacobi 最高阶数")
    trunc_slider = mo.ui.slider(start=0.01, stop=0.30, step=0.01, value=0.05, label="截断阈值比例")

    controls = mo.vstack(
        [
            mo.md("**数据与核参数** "),
            mo.hstack([noise_slider, center_slider, width_slider], justify="space-between"),
            mo.md("**GP / 基函数参数** "),
            mo.hstack([length_slider, order_slider, trunc_slider], justify="space-between"),
        ]
    )
    controls
    return center_slider, controls, length_slider, noise_slider, order_slider, trunc_slider, width_slider


@app.cell
def _(
    ConstantKernel,
    GaussianProcessRegressor,
    RBF,
    center_slider,
    length_slider,
    noise_slider,
    np,
    width_slider,
):
    """①-④ 模拟数据 → GP 拟合 → 后验导数 → 漂移强度驱动超参数演化"""

    np.random.seed(42)
    n_points = 200
    t = np.linspace(0, 10, n_points).reshape(-1, 1)

    def sigmoid(x, center, width):
        return 1 / (1 + np.exp(-(x - center) / width))

    noise_std = noise_slider.value
    night_market = 1 - sigmoid(t.flatten(), center_slider.value, width_slider.value)
    convenience = sigmoid(t.flatten(), center_slider.value, width_slider.value)

    y_night = night_market + np.random.normal(0, noise_std, n_points)
    y_conv = convenience + np.random.normal(0, noise_std, n_points)

    kernel = ConstantKernel(1.0) * RBF(length_scale=length_slider.value)
    gp_night = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=5)
    gp_conv = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=5)
    gp_night.fit(t, y_night)
    gp_conv.fit(t, y_conv)

    t_test = np.linspace(0, 10, 500).reshape(-1, 1)
    mu_night, sigma_night = gp_night.predict(t_test, return_std=True)
    mu_conv, sigma_conv = gp_conv.predict(t_test, return_std=True)

    # 后验导数 → 综合漂移强度
    dt = t_test[1, 0] - t_test[0, 0]
    d_mu_night = np.gradient(mu_night, dt)
    d_mu_conv = np.gradient(mu_conv, dt)
    drift_magnitude = np.sqrt(d_mu_night**2 + d_mu_conv**2)
    drift_norm = (drift_magnitude - drift_magnitude.min()) / (
        drift_magnitude.max() - drift_magnitude.min()
    )

    # 累积漂移强度作为"演化里程"，驱动 α(t), β(t)
    cumulative_drift = np.cumsum(drift_norm) * dt
    cumulative_drift_norm = cumulative_drift / cumulative_drift.max()
    alpha_t = 2.0 * cumulative_drift_norm
    beta_t = 1.5 * cumulative_drift_norm

    return (
        alpha_t,
        beta_t,
        d_mu_night,
        dt,
        drift_magnitude,
        mu_conv,
        mu_night,
        sigma_conv,
        sigma_night,
        t,
        t_test,
        y_conv,
        y_night,
    )


@app.cell
def _(
    alpha_t,
    beta_t,
    jacobi,
    mo,
    mu_night,
    np,
    order_slider,
    t_test,
    trunc_slider,
):
    """⑤-⑥ Jacobi 基展开（自适应超参数）→ 截断剪枝"""

    def jacobi_basis(x, n, alpha, beta):
        return jacobi(n, alpha, beta)(x)

    n_basis = int(order_slider.value) + 1
    coefficients_night = np.zeros((len(t_test), n_basis))
    x_pts = np.linspace(-1, 1, 200)

    for idx, _t_val in enumerate(t_test.flatten()):
        a, b = alpha_t[idx], beta_t[idx]
        signal_val = mu_night[idx]
        weight = (1 - x_pts) ** a * (1 + x_pts) ** b
        signal_vals = signal_val * np.ones_like(x_pts)
        for n in range(n_basis):
            basis_vals = jacobi_basis(x_pts, n, a, b)
            numerator = np.trapezoid(signal_vals * basis_vals * weight, x_pts)
            denominator = np.trapezoid(basis_vals**2 * weight, x_pts)
            coefficients_night[idx, n] = numerator / denominator if denominator > 1e-10 else 0

    mean_coeff_magnitude = np.mean(np.abs(coefficients_night), axis=0)
    truncation_threshold = trunc_slider.value * np.max(mean_coeff_magnitude)
    kept_bases = mean_coeff_magnitude > truncation_threshold
    n_kept = int(np.sum(kept_bases))

    truncation_table = mo.ui.table(
        {
            "order": list(range(n_basis)),
            "mean |coeff|": [round(float(v), 6) for v in mean_coeff_magnitude],
            "status": ["✅ KEPT" if k else "✂️ PRUNED" for k in kept_bases],
        }
    )

    return (
        jacobi_basis,
        kept_bases,
        mean_coeff_magnitude,
        n_basis,
        n_kept,
        truncation_table,
        truncation_threshold,
    )


@app.cell
def _(
    alpha_t,
    beta_t,
    drift_magnitude,
    jacobi_basis,
    kept_bases,
    mean_coeff_magnitude,
    mu_conv,
    mu_night,
    n_basis,
    n_kept,
    np,
    plt,
    sigma_conv,
    sigma_night,
    t,
    t_test,
    truncation_threshold,
    y_conv,
    y_night,
):
    """⑦ 五联可视化"""

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # 子图1：概念空间演化
    axes[0].plot(t_test, mu_night, "r-", label="Night market (GP posterior mean)", linewidth=2)
    axes[0].fill_between(
        t_test.flatten(),
        mu_night - 1.96 * sigma_night,
        mu_night + 1.96 * sigma_night,
        alpha=0.2,
        color="r",
        label="95% confidence interval",
    )
    axes[0].plot(t_test, mu_conv, "b-", label="Convenience store (GP posterior mean)", linewidth=2)
    axes[0].fill_between(
        t_test.flatten(),
        mu_conv - 1.96 * sigma_conv,
        mu_conv + 1.96 * sigma_conv,
        alpha=0.2,
        color="b",
    )
    axes[0].scatter(t.flatten(), y_night, c="r", s=10, alpha=0.3)
    axes[0].scatter(t.flatten(), y_conv, c="b", s=10, alpha=0.3)
    axes[0].set_ylabel("Activity level")
    axes[0].set_title("(1) Concept-space evolution over time (GP posterior)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 子图2：漂移强度
    axes[1].plot(t_test, drift_magnitude, "g-", linewidth=2)
    threshold_val = np.mean(drift_magnitude) + 2 * np.std(drift_magnitude)
    axes[1].axhline(
        y=threshold_val,
        color="orange",
        linestyle="--",
        label=f"Sudden-change threshold ({threshold_val:.3f})",
    )
    axes[1].set_ylabel("Drift intensity")
    axes[1].set_title("(2) Drift intensity diagnosed from GP posterior derivatives")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 子图3：超参数演化
    axes[2].plot(t_test, alpha_t, "m-", label="alpha(t)", linewidth=2)
    axes[2].plot(t_test, beta_t, "c-", label="beta(t)", linewidth=2)
    axes[2].set_ylabel("Hyperparameter value")
    axes[2].set_title("(3) Adaptive hyperparameter evolution (driven by drift intensity)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # 子图4：基函数形变示例
    sample_indices = [50, 250, 450]
    colors_sample = ["blue", "green", "red"]
    labels_sample = ["early t", "mid t", "late t"]
    x_plot = np.linspace(-1, 1, 200)
    for s_idx, s_color, s_label in zip(sample_indices, colors_sample, labels_sample):
        s_a, s_b = alpha_t[s_idx], beta_t[s_idx]
        axes[3].plot(
            x_plot,
            jacobi_basis(x_plot, 2, s_a, s_b),
            color=s_color,
            linewidth=2,
            label=f"{s_label}: alpha={s_a:.2f}, beta={s_b:.2f}",
        )
    axes[3].set_ylabel("Basis function value")
    axes[3].set_title("(4) Jacobi basis-function deformation (order-2 polynomial)")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # 子图5：截断策略
    bar_colors = ["green" if kept else "red" for kept in kept_bases]
    axes[4].bar(range(n_basis), mean_coeff_magnitude, color=bar_colors, alpha=0.7)
    axes[4].axhline(
        y=truncation_threshold,
        color="orange",
        linestyle="--",
        label=f"Truncation threshold ({truncation_threshold:.6f})",
    )
    axes[4].set_xlabel("Basis function order")
    axes[4].set_ylabel("Mean coefficient magnitude")
    axes[4].set_title(
        f"(5) Truncation strategy: kept {n_kept}/{n_basis} basis functions (green=kept, red=pruned)"
    )
    axes[4].set_xticks(range(n_basis))
    axes[4].legend(fontsize=8)
    axes[4].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plt.close(fig)
    fig
    return fig, threshold_val


@app.cell
def _(
    alpha_t,
    beta_t,
    d_mu_night,
    mo,
    n_basis,
    n_kept,
    np,
    threshold_val,
    truncation_table,
):
    """⑧ 诊断结论"""

    max_deriv = np.max(np.abs(d_mu_night))
    mean_deriv = np.mean(np.abs(d_mu_night))
    threshold_deriv = mean_deriv + 2 * np.std(np.abs(d_mu_night))

    if max_deriv > threshold_deriv * 1.5:
        verdict = "🚨 **检测到突变漂移（Sudden drift）** —— 导数出现尖峰"
    else:
        verdict = (
            "🌊 **渐进漂移（Gradual drift）** —— 导数平缓、无尖峰；"
            "超参数已按漂移强度自适应演化，基函数空间连续形变"
        )

    mo.md(
        f"""
        ### 📊 漂移类型诊断

        | 指标 | 数值 |
        |---|---|
        | 夜市导数最大值 | `{max_deriv:.4f}` |
        | 夜市导数均值 | `{mean_deriv:.4f}` |
        | 突变阈值（均值 + 2σ） | `{threshold_deriv:.4f}` |
        | 漂移强度阈值（子图2） | `{threshold_val:.4f}` |
        | 超参数 α 演化范围 | `[{alpha_t.min():.3f}, {alpha_t.max():.3f}]` |
        | 超参数 β 演化范围 | `[{beta_t.min():.3f}, {beta_t.max():.3f}]` |

        {verdict}

        ### ✂️ 截断策略结果（保留 {n_kept} / {n_basis} 个基函数）

        {truncation_table}
        """
    )


if __name__ == "__main__":
    app.run()
