# ============================================================
# gp-diagnoistoc_v2.py  (improved copy — original left untouched)
#
# Fixes vs v1:
#   1. NO circular truncation experiment: we now expand the FULL
#      posterior trajectory mu(t) in a Legendre/Jacobi basis and
#      quantify truncation by reconstruction RMSE (falsifiable).
#   2. Principled drift-type detection: Bayesian model comparison
#      (log marginal likelihood / Bayes factor) between a SMOOTH
#      kernel hypothesis and an ABRUPT kernel hypothesis, instead
#      of the ad-hoc mean+2*std threshold.
#   3. Stress test on three ground-truth regimes (gradual sigmoid,
#      step, linear ramp) -> confusion table, so the detector can
#      actually fail and show it.
#   4. No name clash: GP noise term is `noise_var`, Jacobi weight
#      exponents are `jac_alpha, jac_beta`.
# ============================================================
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.special import jacobi
from sklearn.exceptions import ConvergenceWarning

# length scales are deliberately PINNED to hypothesis values for model
# comparison, so boundary-convergence notices are expected -> silence them
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
N_POINTS = 200
NOISE_STD = 0.05
noise_var = NOISE_STD**2

T = np.linspace(0, 10, N_POINTS)
X = T.reshape(-1, 1)

# ------------------------------------------------------------
# 1. Three ground-truth drift regimes (the falsifiable part)
# ------------------------------------------------------------
def sigmoid(x, center=5, width=1.5):
    return 1 / (1 + np.exp(-(x - center) / width))

truth = {
    "gradual (sigmoid w=1.5)": 1 - sigmoid(T, 5, 1.5),
    "step    (sigmoid w=0.05)": 1 - sigmoid(T, 5, 0.05),
    "ramp    (linear 3->7)":    np.clip(1 - (T - 3) / 4, 0, 1),
}
expected = {  # what the detector should conclude
    "gradual (sigmoid w=1.5)": "GRADUAL",
    "step    (sigmoid w=0.05)": "ABRUPT",
    # NOTE: a ramp has DISCONTINUOUS derivatives at its two kinks, so an
    # "ABRUPT" verdict there is arguably defensible, not a pure error.
    "ramp    (linear 3->7)":   "GRADUAL",
}

# ------------------------------------------------------------
# 2. Two competing hypotheses as GP kernels
#    H_smooth: long length scale -> drift spread over time
#    H_abrupt: short length scale -> drift concentrated in a spike
# ------------------------------------------------------------
def fit_gp(y, length_scale):
    # length scale FIXED to the hypothesis value (tight bounds) so the comparison
    # is a genuine model comparison, not two re-tuned copies
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale, (0.99 * length_scale, 1.01 * length_scale)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=noise_var, normalize_y=True, n_restarts_optimizer=5
    )
    gp.fit(X, y)
    lml = gp.log_marginal_likelihood(gp.kernel_.theta)
    return gp, lml

T_TEST = np.linspace(0, 10, 500).reshape(-1, 1)
dt = T_TEST[1, 0] - T_TEST[0, 0]

results = {}
print("=" * 78)
print("Bayesian drift-type diagnosis (log marginal likelihood, Bayes factor)")
print("=" * 78)

for name, clean in truth.items():
    np.random.seed(SEED)
    y = clean + np.random.normal(0, NOISE_STD, N_POINTS)

    gp_s, lml_s = fit_gp(y, length_scale=3.0)   # H_smooth
    gp_a, lml_a = fit_gp(y, length_scale=0.3)   # H_abrupt

    mu_s, sd_s = gp_s.predict(T_TEST, return_std=True)
    mu_a, sd_a = gp_a.predict(T_TEST, return_std=True)

    # Bayes factor in favour of the ABRUPT hypothesis
    log_bf = lml_a - lml_s
    posterior_abrupt = 1 / (1 + np.exp(-log_bf))  # equal 50/50 prior
    verdict = "ABRUPT" if posterior_abrupt > 0.5 else "GRADUAL"
    mu_win = mu_a if posterior_abrupt > 0.5 else mu_s  # winner used for expansion

    ok = "OK " if verdict == expected[name] else "MISS"
    results[name] = dict(
        y=y, mu_s=mu_s, sd_s=sd_s, mu_a=mu_a, sd_a=sd_a, mu_win=mu_win,
        lml_s=lml_s, lml_a=lml_a, post_abrupt=posterior_abrupt,
        verdict=verdict, expected=expected[name], ok=ok,
    )
    print(f"  {name:28s}  LML smooth={lml_s:8.2f}  abrupt={lml_a:8.2f}"
          f"  P(abrupt|D)={posterior_abrupt:6.4f}  -> {verdict:8s} [{ok}]")

n_correct = sum(r["verdict"] == r["expected"] for r in results.values())
print(f"\n  Accuracy on held-out regimes: {n_correct}/{len(results)}")

# ------------------------------------------------------------
# 3. REAL Jacobi/Legendre expansion of the posterior trajectory
#    (not the constant-signal pseudo-experiment from v1)
#    coefficients c_n = <mu, P_n>_w / <P_n, P_n>_w over x in [-1,1]
# ------------------------------------------------------------
jac_alpha, jac_beta = 0.0, 0.0            # Legendre (symmetric) weighting
MAX_ORDER = 15
x_pts = np.linspace(-1, 1, 800)

def jacobi_basis(x, n, a, b):
    return jacobi(n, a, b)(x)

def weight(x, a, b):
    return np.clip(1 - x, 1e-12, None) ** a * np.clip(1 + x, 1e-12, None) ** b

print()
print("=" * 78)
print("Spectral truncation with QUANTIFIED reconstruction error")
print("=" * 78)

trunc_report = {}
for name, r in results.items():
    mu = r["mu_win"]                      # posterior of the winning hypothesis
    # map time axis [0,10] -> x in [-1,1] and interpolate posterior onto x grid
    mu_on_x = np.interp(x_pts, (T_TEST.flatten() - 5) / 5, mu)

    coeffs = np.zeros(MAX_ORDER + 1)
    for n in range(MAX_ORDER + 1):
        P = jacobi_basis(x_pts, n, jac_alpha, jac_beta)
        w = weight(x_pts, jac_alpha, jac_beta)
        coeffs[n] = np.trapezoid(mu_on_x * P * w, x_pts) / np.trapezoid(P**2 * w, x_pts)

    # cumulative reconstruction + RMSE as we keep more basis functions
    recon = np.zeros_like(x_pts)
    rmse_curve = np.zeros(MAX_ORDER + 1)
    for n in range(MAX_ORDER + 1):
        recon += coeffs[n] * jacobi_basis(x_pts, n, jac_alpha, jac_beta)
        rmse_curve[n] = np.sqrt(np.mean((recon - mu_on_x) ** 2))

    # prune: smallest # of (largest-|c|) terms with RMSE < 2% of signal std
    tolerance = 0.02 * np.std(mu_on_x)
    order_idx = np.argsort(-np.abs(coeffs))
    n_needed = MAX_ORDER + 1
    for k in range(1, MAX_ORDER + 2):
        sel = np.sort(order_idx[:k])
        rec_k = sum(coeffs[j] * jacobi_basis(x_pts, j, jac_alpha, jac_beta) for j in sel)
        if np.sqrt(np.mean((rec_k - mu_on_x) ** 2)) < tolerance:
            n_needed = k
            break

    trunc_report[name] = dict(coeffs=coeffs, rmse_curve=rmse_curve,
                              n_needed=n_needed, tolerance=tolerance)
    top = [int(j) for j in np.sort(order_idx[: min(n_needed, 6)])]
    print(f"  {name:28s}  kept {n_needed:2d}/{MAX_ORDER+1} terms for RMSE < 2% of signal std"
          f"   (top orders: {top})")

# ------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex="col")

for row, (name, r) in enumerate(results.items()):
    ax = axes[row, 0]
    ax.scatter(T, r["y"], c="k", s=8, alpha=0.3, label="noisy data")
    ax.plot(T_TEST, r["mu_s"], "b-", lw=2, label=f"H_smooth  (LML={r['lml_s']:.1f})")
    ax.fill_between(T_TEST.flatten(), r["mu_s"] - 2 * r["sd_s"], r["mu_s"] + 2 * r["sd_s"],
                    alpha=0.15, color="b")
    ax.plot(T_TEST, r["mu_a"], "r-", lw=2, label=f"H_abrupt  (LML={r['lml_a']:.1f})")
    ax.fill_between(T_TEST.flatten(), r["mu_a"] - 2 * r["sd_a"], r["mu_a"] + 2 * r["sd_a"],
                    alpha=0.15, color="r")
    ax.set_ylabel("activity")
    ax.set_title(f"truth: {name}  |  verdict: {r['verdict']} ({r['ok']}), P(abrupt)={r['post_abrupt']:.3f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[row, 1]
    rep = trunc_report[name]
    ax.bar(range(MAX_ORDER + 1), np.abs(rep["coeffs"]), color="steelblue", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("basis order n")
    ax.set_ylabel("|c_n|")
    ax.set_title("Jacobi spectrum of posterior trajectory")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[row, 2]
    ax.plot(range(MAX_ORDER + 1), rep["rmse_curve"], "go-", label="cumulative RMSE")
    ax.axhline(rep["tolerance"], color="orange", ls="--", label="2% of signal std")
    ax.axvline(rep["n_needed"] - 1, color="gray", ls=":", label=f"kept={rep['n_needed']}")
    ax.set_yscale("log")
    ax.set_xlabel("number of terms kept")
    ax.set_ylabel("reconstruction RMSE")
    ax.set_title("Truncation cost is measurable (falsifiable)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gp_diag_v2.png", dpi=150, bbox_inches="tight")
print("\nSaved plot to gp_diag_v2.png")

# ------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------
print()
print("=" * 78)
print("What changed vs v1")
print("=" * 78)
print("""  - Detection: Bayesian model comparison (log marginal likelihood,
    posterior P(abrupt|D)) replaces the ad-hoc mean+2*sigma threshold.
  - Validation: three ground-truth regimes; the detector can now miss.
  - Truncation: expansion of the real posterior trajectory mu(t);
    pruning justified by measured reconstruction RMSE, not by a
    constant-signal tautology.
  - Naming: GP noise_var is separated from Jacobi exponents jac_alpha/jac_beta.""")
