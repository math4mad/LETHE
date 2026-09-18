import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.special import jacobi

# ============================================================
# 1. Generate simulated data: gradual drift from Night Market -> Convenience Store
# ============================================================
np.random.seed(42)
n_points = 200
t = np.linspace(0, 10, n_points).reshape(-1, 1)  # time axis

# True drift: sigmoid simulates a smooth transition
def sigmoid(x, center=5, width=1.5):
    return 1 / (1 + np.exp(-(x - center) / width))

night_market = 1 - sigmoid(t.flatten())   # night-market objects gradually decrease
convenience = sigmoid(t.flatten())         # convenience-store objects gradually increase

# Add observation noise
noise_std = 0.05
y_night = night_market + np.random.normal(0, noise_std, n_points)
y_conv = convenience + np.random.normal(0, noise_std, n_points)

# ============================================================
# 2. GP fitting
# ============================================================
kernel = ConstantKernel(1.0) * RBF(length_scale=2.0)
gp_night = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=5)
gp_conv = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=5)

gp_night.fit(t, y_night)
gp_conv.fit(t, y_conv)

# Predict posterior mean and standard deviation
t_test = np.linspace(0, 10, 500).reshape(-1, 1)
mu_night, sigma_night = gp_night.predict(t_test, return_std=True)
mu_conv, sigma_conv = gp_conv.predict(t_test, return_std=True)

# ============================================================
# 3. Compute posterior derivatives (diagnose drift intensity)
# ============================================================
dt = t_test[1, 0] - t_test[0, 0]
d_mu_night = np.gradient(mu_night, dt)  # first derivative
d_mu_conv = np.gradient(mu_conv, dt)

# Combined drift magnitude
drift_magnitude = np.sqrt(d_mu_night**2 + d_mu_conv**2)

# Normalize to [0, 1]
drift_norm = (drift_magnitude - drift_magnitude.min()) / (drift_magnitude.max() - drift_magnitude.min())

# ============================================================
# 4. Adaptive hyperparameter evolution: drift intensity drives alpha(t), beta(t)
# ============================================================
# Initial hyperparameters (Legendre basis, symmetric weights)
alpha_0, beta_0 = 0.0, 0.0

# Cumulative drift intensity as "evolution mileage"
cumulative_drift = np.cumsum(drift_norm) * dt
cumulative_drift_norm = cumulative_drift / cumulative_drift.max()

# Hyperparameter evolution trajectories
alpha_t = alpha_0 + 2.0 * cumulative_drift_norm   # alpha from 0 -> 2
beta_t = beta_0 + 1.5 * cumulative_drift_norm      # beta from 0 -> 1.5

# ============================================================
# 5. Build Jacobi basis with adaptive hyperparameters, expand night-market signal
# ============================================================
def jacobi_basis(x, n, alpha, beta):
    """Evaluate the n-th order Jacobi polynomial at points x"""
    P = jacobi(n, alpha, beta)
    return P(x)

# Highest basis-function order
max_order = 5
n_basis = max_order + 1

# For each time point, compute Jacobi basis expansion coefficients
coefficients_night = np.zeros((len(t_test), n_basis))

# x points for numerical integration
x_pts = np.linspace(-1, 1, 200)

for idx, t_val in enumerate(t_test.flatten()):
    a = alpha_t[idx]
    b = beta_t[idx]
    
    # Signal value at current time point (approximated by GP posterior mean)
    signal_val = mu_night[idx]
    
    for n in range(n_basis):
        # Compute basis function values
        basis_vals = jacobi_basis(x_pts, n, a, b)
        
        # Weight function w(x) = (1-x)^alpha (1+x)^beta
        weight = (1 - x_pts)**a * (1 + x_pts)**b
        
        # Signal approximated as constant (simplification)
        signal_vals = signal_val * np.ones_like(x_pts)
        
        # Coefficient = <signal, basis>_w / <basis, basis>_w
        numerator = np.trapezoid(signal_vals * basis_vals * weight, x_pts)
        denominator = np.trapezoid(basis_vals**2 * weight, x_pts)
        
        coefficients_night[idx, n] = numerator / denominator if denominator > 1e-10 else 0

# ============================================================
# 6. Truncation strategy: automatically prune redundant basis functions
# ============================================================
# Mean coefficient magnitude of each basis function over the whole time axis
mean_coeff_magnitude = np.mean(np.abs(coefficients_night), axis=0)

# Truncation threshold: coefficient magnitude below 5% of the max is redundant
truncation_threshold = 0.05 * np.max(mean_coeff_magnitude)

# Mark which basis functions are kept
kept_bases = mean_coeff_magnitude > truncation_threshold
n_kept = np.sum(kept_bases)

print("=" * 50)
print("Truncation strategy results:")
print("=" * 50)
for n in range(n_basis):
    status = "KEPT" if kept_bases[n] else "PRUNED"
    print(f"  Order-{n} basis: mean coeff magnitude = {mean_coeff_magnitude[n]:.6f}  ->  {status}")
print(f"\nBasis functions kept: {n_kept} / {n_basis}")

# Reconstruct signal using kept basis functions
coefficients_truncated = coefficients_night.copy()
coefficients_truncated[:, ~kept_bases] = 0  # zero out pruned coefficients

# Reconstruct signal (simplified: only order-0 basis used since signal is approx constant)
reconstructed_signal = coefficients_truncated[:, 0] * jacobi_basis(0, 0, alpha_t, beta_t)

# ============================================================
# 7. Visualization
# ============================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

# Subplot 1: concept-space evolution
axes[0].plot(t_test, mu_night, 'r-', label='Night market (GP posterior mean)', linewidth=2)
axes[0].fill_between(t_test.flatten(),
                     mu_night - 1.96*sigma_night,
                     mu_night + 1.96*sigma_night,
                     alpha=0.2, color='r', label='95% confidence interval')
axes[0].plot(t_test, mu_conv, 'b-', label='Convenience store (GP posterior mean)', linewidth=2)
axes[0].fill_between(t_test.flatten(),
                     mu_conv - 1.96*sigma_conv,
                     mu_conv + 1.96*sigma_conv,
                     alpha=0.2, color='b')
axes[0].scatter(t.flatten(), y_night, c='r', s=10, alpha=0.3)
axes[0].scatter(t.flatten(), y_conv, c='b', s=10, alpha=0.3)
axes[0].set_ylabel('Activity level')
axes[0].set_title('(1) Concept-space evolution over time (GP posterior)')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(True, alpha=0.3)

# Subplot 2: drift intensity
axes[1].plot(t_test, drift_magnitude, 'g-', linewidth=2)
threshold_val = np.mean(drift_magnitude) + 2 * np.std(drift_magnitude)
axes[1].axhline(y=threshold_val, color='orange', linestyle='--', label=f'Sudden-change threshold ({threshold_val:.3f})')
axes[1].set_ylabel('Drift intensity')
axes[1].set_title('(2) Drift intensity diagnosed from GP posterior derivatives')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Subplot 3: hyperparameter evolution
axes[2].plot(t_test, alpha_t, 'm-', label='alpha(t)', linewidth=2)
axes[2].plot(t_test, beta_t, 'c-', label='beta(t)', linewidth=2)
axes[2].set_ylabel('Hyperparameter value')
axes[2].set_title('(3) Adaptive hyperparameter evolution (driven by drift intensity)')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

# Subplot 4: basis-function deformation examples
sample_indices = [50, 250, 450]
colors_sample = ['blue', 'green', 'red']
labels_sample = ['early t', 'mid t', 'late t']

x_plot = np.linspace(-1, 1, 200)
for idx, color, label in zip(sample_indices, colors_sample, labels_sample):
    a = alpha_t[idx]
    b = beta_t[idx]
    P2 = jacobi_basis(x_plot, 2, a, b)
    axes[3].plot(x_plot, P2, color=color, linewidth=2,
                 label=f'{label}: alpha={a:.2f}, beta={b:.2f}')

axes[3].set_ylabel('Basis function value')
axes[3].set_title('(4) Jacobi basis-function deformation (order-2 polynomial)')
axes[3].legend(fontsize=8)
axes[3].grid(True, alpha=0.3)

# Subplot 5: truncation strategy - basis coefficient magnitudes
bar_colors = ['green' if kept else 'red' for kept in kept_bases]
bars = axes[4].bar(range(n_basis), mean_coeff_magnitude, color=bar_colors, alpha=0.7)
axes[4].axhline(y=truncation_threshold, color='orange', linestyle='--',
                label=f'Truncation threshold ({truncation_threshold:.6f})')
axes[4].set_xlabel('Basis function order')
axes[4].set_ylabel('Mean coefficient magnitude')
axes[4].set_title(f'(5) Truncation strategy: kept {n_kept}/{n_basis} basis functions (green=kept, red=pruned)')
axes[4].set_xticks(range(n_basis))
axes[4].legend(fontsize=8)
axes[4].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('full_pipeline_with_truncation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. Print diagnostic information
# ============================================================
max_deriv = np.max(np.abs(d_mu_night))
mean_deriv = np.mean(np.abs(d_mu_night))
threshold_deriv = mean_deriv + 2 * np.std(np.abs(d_mu_night))

print("\n" + "=" * 50)
print("Drift type diagnosis:")
print("=" * 50)
print(f"Night-market derivative max:   {max_deriv:.4f}")
print(f"Night-market derivative mean:  {mean_deriv:.4f}")
print(f"Sudden-change threshold (mean+2*std): {threshold_deriv:.4f}")

if max_deriv > threshold_deriv * 1.5:
    print("-> Sudden (abrupt) drift detected!")
else:
    print("-> Gradual drift (derivatives smooth, no spikes)")
    print("-> Hyperparameters evolved adaptively with drift intensity; basis space deforms continuously")

print(f"\nHyperparameter evolution ranges:")
print(f"  alpha: [{alpha_t.min():.3f}, {alpha_t.max():.3f}]")
print(f"  beta:  [{beta_t.min():.3f}, {beta_t.max():.3f}]")
