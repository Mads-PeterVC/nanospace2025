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
fig, axes = plt.subplots(2, 4, figsize=(4*sz, 2*sz), constrained_layout=True)

for i, length_scale in enumerate([0.001, 0.025, 0.1, 0.25]):

    X = np.linspace(0, 1, 100)
    covariance = kernel(X, X, length_scale=length_scale)
    mean = np.zeros(len(X))
    samples = sample_gp(mean, covariance, num_samples=3)

    ax = axes[0, i]
    linestyle = '-' if add_lines else 'o'
    ax.plot(X, samples, linestyle, label='Sampled GP')
    ax.set_xlabel('$X$')
    ax.set_ylabel(r'$y \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$')
    ax.set_title('GP Samples - $\sigma = {:.3f}$'.format(length_scale))

    ax = axes[1, i]
    ax.imshow(covariance, cmap='Purples', extent=(0, 1, 0, 1), origin='upper',
               aspect='auto', interpolation='nearest')
    ax.axis('equal')
    ax.axis('off')


plt.savefig('figures/gp_sampling_5.png')





