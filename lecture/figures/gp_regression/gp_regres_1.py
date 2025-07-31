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


sz = 3
fig, axes = plt.subplots(1, 1, figsize=(1*sz, 1*sz), constrained_layout=True)
axes = np.atleast_1d(axes)

for i, length_scale in enumerate([0.1]):

    X = np.linspace(0, 1, 100)
    covariance = kernel(X, X, length_scale=length_scale)
    mean = np.zeros(len(X))
    samples = sample_gp(mean, covariance, num_samples=100)

    ax = axes[i]
    linestyle = '-' if add_lines else 'o'
    ax.plot(X, samples, linestyle, label='Sampled GP', color='gray', alpha=0.5, linewidth=0.5)

    ax.set_xlabel('$X$')
    ax.set_ylabel(r'$y \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$')
    ax.set_xlim((0, 1))

plt.savefig('figures/gp_regres_1.png')





