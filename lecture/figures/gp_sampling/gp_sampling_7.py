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

def kernel(x1, x2, length_scale=1.0, v=1/2):
    d = cdist(x1[:, np.newaxis], x2[:, np.newaxis], metric='euclidean')

    if v == 1/2:
        print(v)
        k = np.exp(-d / (2 * length_scale**2))
    elif v == 3/2:
        print(v)
        k = (1 + np.sqrt(3) * d / length_scale) * np.exp(-np.sqrt(3) * d / length_scale)
    elif v == 5/2:
        print(v)
        k = (1 + np.sqrt(5) * d / length_scale + 5 * d**2 / (3 * length_scale**2)) * np.exp(-np.sqrt(5) * d / length_scale)
    return k


sz = 3
fig, axes = plt.subplots(2, 3, figsize=(3*sz, 2*sz), constrained_layout=True)

labels = ['1/16', '1/8', '1/4']

for i, v in enumerate([1/2, 3/2, 5/2]):

    X = np.linspace(0, 1, 100)
    covariance = kernel(X, X, length_scale=0.10, v=v) + 1e-6 * np.eye(len(X))
    mean = np.zeros(len(X))
    samples = sample_gp(mean, covariance, num_samples=3)

    ax = axes[0, i]
    linestyle = '-' if add_lines else 'o'
    ax.plot(X, samples, linestyle, label='Sampled GP')
    ax.set_xlabel('$X$')
    ax.set_ylabel(r'$y \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$')
    ax.set_title(f'$\sigma = 0.1$, $v = {v}$')

    ax = axes[1, i]
    ax.imshow(covariance, cmap='Purples', extent=(0, 1, 0, 1), origin='upper',
               aspect='auto', interpolation='nearest')
    ax.axis('equal')
    ax.axis('off')


plt.savefig('figures/gp_sampling_7.png')





