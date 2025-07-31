import numpy as np
import matplotlib.pyplot as plt
from bayes_nanospace2025.lecture import set_plot_settings
from scipy.spatial.distance import cdist

np.random.seed(42)  # For reproducibility

set_plot_settings()

plt.switch_backend('webagg')

def sample_gp(mean, covariance, num_samples=1):
    return np.random.multivariate_normal(mean, covariance, num_samples).T

add_lines = True

def kernel(x1, x2, length_scale=0.1):
    d = cdist(x1[:, np.newaxis], x2[:, np.newaxis], metric='sqeuclidean')
    return np.exp(-0.5 * d / length_scale**2)


# Prior
X = np.linspace(0, 1, 1000)
covariance = kernel(X, X, length_scale=0.1)
mean = np.zeros(len(X))
samples_prior = sample_gp(mean, covariance, num_samples=100)


# Posterior: 
xm = np.array([0.33, 0.66, 0.99])
ym = np.array([0.2, 1.0, -1])

K_mm = kernel(xm, xm)
K_mm_inv = np.linalg.inv(K_mm + 1e-6 * np.eye(len(xm)))

k_m = kernel(X, xm)
mean_post = k_m @ K_mm_inv @ ym
alpha = K_mm_inv @ ym

cov_post = kernel(X, X) - k_m @ K_mm_inv @ k_m.T

# Sample from posterior
samples_post = sample_gp(mean_post, cov_post, num_samples=100)

sz = 4
fig, axes = plt.subplots(1, 2, figsize=(2*sz, 1*sz), constrained_layout=True, sharey=True)

ax = axes[0]
ax.plot(X, k_m[:, 0], label='k_m', color='C0', linewidth=2.0)
ax.plot(X, k_m[:, 1], label='k_m', color='C1', linewidth=2.0)
ax.plot(X, k_m[:, 2], label='k_m', color='C2', linewidth=2.0)

ax = axes[1]
ax.plot(X, k_m[:, 0]*alpha[0], label='k_m', color='C0', linewidth=2.0)
ax.plot(X, k_m[:, 1]*alpha[1], label='k_m', color='C1', linewidth=2.0)
ax.plot(X, k_m[:, 2]*alpha[2], label='k_m', color='C2', linewidth=2.0)

ax.plot(X, mean_post, label='Posterior Mean', color='black', linewidth=3, zorder=-2)
ax.plot(xm, ym, 'o', label='Target Points', color='red', markersize=5, zorder=10)

for i, (x, a) in enumerate(zip(xm, alpha)):
    ax.plot([x, x], [0, a], '-', linewidth=2.0, alpha=0.5, color=f'C{i}')


ax.plot(X, samples_post, '-', label='Posterior Samples', color='gray', alpha=0.1, linewidth=1.0)

axes[0].set_xlabel('$X$')
axes[1].set_xlabel('$X$')
axes[0].set_ylabel(r'$y$')


for ax in axes:
    ax.set_xlim((0, 1))
    ax.set_ylim((-3, 3)) 







plt.savefig('figures/gp_regres_4.png', dpi=300)








