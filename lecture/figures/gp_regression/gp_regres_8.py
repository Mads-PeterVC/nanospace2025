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

def kernel(x1, x2, length_scale=1.0, p=0.1):
    d = cdist(x1[:, np.newaxis], x2[:, np.newaxis], metric='euclidean')
    k = np.exp(-2 * np.sin(np.pi * d/p)**2 / length_scale**2)
    return k

# Posterior: 
xm = np.array([0.33, 0.66, 0.99])
ym = np.array([0.2, 1.0, -1])
X = np.linspace(0, 1, 5000)


def sample_posterior(noise=1e-3, length_scale=0.1, p=0.1):
    K_mm = kernel(xm, xm, length_scale=length_scale, p=p)
    K_mm_inv = np.linalg.inv(K_mm + noise * np.eye(len(xm)))

    k_m = kernel(X, xm, length_scale=length_scale, p=p)
    mean_post = k_m @ K_mm_inv @ ym
    cov_post = kernel(X, X, length_scale, p=p) - k_m @ K_mm_inv @ k_m.T

    # Sample from posterior
    samples_post = None

    return mean_post, samples_post, cov_post

sz = 4
fig, ax = plt.subplots(1, 3, figsize=(3*sz, 1*sz), constrained_layout=True)

labels = [r'p=1/16\pi', r'p=1/8\pi', r'p=1/4\pi']
for i, p in enumerate([1/16*np.pi, 1/8*np.pi, 1/4*np.pi]):

    mean_post, samples_post, cov_post = sample_posterior(noise=1e-6, length_scale=0.25, p=p)

    var = np.diag(cov_post)
    std = np.sqrt(var)

    # Plot posterior samples
    ax[i].plot(xm, ym, 'o', label='Target Points', color='red', markersize=5, zorder=10)
    ax[i].fill_between(X, mean_post - 2*std, mean_post + 2*std, color='lightgray', alpha=1.0, label='Posterior Variance')
        
    ax[i].plot(X, mean_post, label='Posterior Mean', color='black', linewidth=2)
    ax[i].set_title(f'${labels[i]}$')
    ax[i].set_xlabel('$X$')
    ax[i].set_xlim((0, 1))
    ax[i].set_ylim((-3, 3))


plt.savefig('figures/gp_regres_8.png', dpi=300)








