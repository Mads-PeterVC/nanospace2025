import numpy as np
import matplotlib.pyplot as plt
from bayes_nanospace2025.lecture import set_plot_settings
from scipy.spatial.distance import cdist
from matplotlib import cm, ticker
np.random.seed(42)  # For reproducibility

set_plot_settings()

plt.switch_backend("webagg")


def sample_gp(mean, covariance, num_samples=1):
    return np.random.multivariate_normal(mean, covariance, num_samples).T

def kernel(x1, x2, length_scale=0.1):
    d = cdist(x1[:, np.newaxis], x2[:, np.newaxis], metric="sqeuclidean")
    return np.exp(-0.5 * d / length_scale**2)

def sample_posterior(X, xm, ym, noise=1e-3, length_scale=0.1):
    K_mm = kernel(xm, xm, length_scale=length_scale)
    K_mm_inv = np.linalg.inv(K_mm + noise * np.eye(len(xm)))

    k_m = kernel(X, xm, length_scale=length_scale)
    mean_post = k_m @ K_mm_inv @ ym
    cov_post = kernel(X, X, length_scale) - k_m @ K_mm_inv @ k_m.T

    return mean_post, cov_post

def compute_lml(xm, ym, noise=1e-3, length_scale=0.1):
    K_mm = kernel(xm, xm, length_scale=length_scale)

    L = np.linalg.cholesky(K_mm + noise * np.eye(len(xm)))
    L_inv = np.linalg.inv(L)
    K_mm_inv = L_inv.T @ L_inv

    logdet = 2 * np.sum(np.log(np.diag(L)))

    lml = (
        -0.5 * ym.T @ K_mm_inv @ ym
        - 0.5 * logdet
        - len(xm) / 2 * np.log(2 * np.pi)
    )
    return lml

xm = np.array([0.33, 0.66, 0.99, 0.68])
ym = np.array([0.2, 1.0, -1, 2])


sz = 4
fig, ax = plt.subplots(1, 1, figsize=(1.5 * sz, 1 * sz), constrained_layout=True)

noise_values = np.linspace(1e-3, 2, 100)
length_scale_values = np.linspace(1e-2, 2.5, 1000)

L, N = np.meshgrid(length_scale_values, noise_values)
lml_values = np.zeros(L.shape)

for i in range(len(noise_values)):
    for j in range(len(length_scale_values)):
        lml_values[i, j] = compute_lml(xm, ym, noise=noise_values[i], length_scale=length_scale_values[j])

index = np.unravel_index(np.argmax(lml_values), lml_values.shape)
lml_values[lml_values < -25] = None

n_levels = 30
c = ax.contour(L, N, lml_values, levels=n_levels, colors='black', linewidths=0.5)
c = ax.contourf(L, N, lml_values, levels=n_levels, cmap="Purples")
fig.colorbar(c, ax=ax, label="Log Marginal Likelihood")
ax.set_xlabel("Length Scale")
ax.set_ylabel("Noise")
ax.set_title("Log Marginal Likelihood")
ax.set_xscale("log")
ax.set_yscale("log")

best_length_scale = length_scale_values[index[1]]
best_noise = noise_values[index[0]]
ax.plot(best_length_scale, best_noise, "bo", markersize=8, label=f"Best Length Scale: {best_length_scale:.2f}, Best Noise: {best_noise:.2f}")

# Picked parameters
length_scale = [0.05, 1.5, 0.5]
noise = [0.005, 1.0, 1e-3]

for ls, n in zip(length_scale, noise):
    ax.plot(ls, n, "ro", markersize=8, label=f"Length Scale: {ls}, Noise: {n}")

plt.savefig("figures/gp_lml.png", dpi=300, bbox_inches="tight")

length_scale.append(best_length_scale)
noise.append(best_noise)

for i, (ls, n) in enumerate(zip(length_scale, noise)):

    fig, ax = plt.subplots(1, 1, figsize=(1 * sz, 1 * sz), constrained_layout=True)

    mean_post, cov_post = sample_posterior(np.linspace(0, 1, 1000), xm, ym, noise=n, length_scale=ls)
    var = np.diag(cov_post)
    std = np.sqrt(var)

    ax.plot(xm, ym, 'o', label='Target Points', color='red', markersize=5, zorder=10)
    ax.plot(np.linspace(0, 1, 1000), mean_post)
    ax.fill_between(np.linspace(0, 1, 1000), mean_post - 2 * std, mean_post + 2 * std, color='lightgray', alpha=0.5)

    ax.set_xlabel('$X$')
    ax.set_xlim((0, 1))
    ax.set_ylim((-3, 3))
    # ax.set_title(f'Posterior with Length Scale: {ls}, Noise: {n}')
    ax.legend()
    plt.savefig(f'figures/gp_lml_{i}.png', dpi=300, bbox_inches='tight')



