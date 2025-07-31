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

def kernel(x1, x2, length_scale=1.0):
    d = cdist(x1[:, np.newaxis], x2[:, np.newaxis], metric='sqeuclidean')
    return np.exp(-0.5 * d / length_scale**2)

x_targets = np.array([0.33, 0.66, 0.99])
y_targets = np.array([0.2, 1.0, -1])


for i, n_targets in enumerate([0, 1, 2, 3]):

    sz = 3
    fig, axes = plt.subplots(1, 1, figsize=(1*sz, 1*sz), constrained_layout=True)
    axes = np.atleast_1d(axes)

    x_target = x_targets[:n_targets]
    y_target = y_targets[:n_targets]

    X = np.linspace(0, 1, 1000)


    axes[0].plot(x_target, y_target, 'o', label='Target points', color='red', markersize=2, zorder=100)

    covariance = kernel(X, X, length_scale=0.1)
    mean = np.zeros(len(X))
    samples = sample_gp(mean, covariance, num_samples=10000)


    x_idx = np.argmin(((X[:, np.newaxis]-x_target[np.newaxis, :]))**2, axis=0).flatten()
    l = np.sum((samples[x_idx, :].squeeze() - y_target[:, np.newaxis])**2, axis=0)
    good_idx = np.argwhere(l < 0.05).flatten()

    
    ax = axes[0]
    linestyle = '-' if add_lines else 'o'
    ax.plot(X, samples[:, good_idx], linestyle, label='Sampled GP', color='gray', alpha=0.5, linewidth=0.5)
    ax.plot(X, samples[:, good_idx].mean(axis=1), linestyle, label='Sampled GP', color='black', alpha=0.75, linewidth=1.5)


    ax.set_xlabel('$X$')
    ax.set_ylabel(r'$y \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$')
    ax.set_title(f'Samples: {len(good_idx)}')
    ax.set_xlim((0, 1))

    plt.savefig(f'figures/gp_regres_2_ntarget{n_targets}.png')





