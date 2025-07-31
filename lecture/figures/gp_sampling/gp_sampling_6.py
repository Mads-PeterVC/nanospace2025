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


sz = 3
fig, axes = plt.subplots(2, 3, figsize=(3*sz, 2*sz), constrained_layout=True)

labels = ['1/16', '1/8', '1/4']

for i, p in enumerate([np.pi/16, np.pi/8, np.pi/4]):

    X = np.linspace(0, 1, 100)
    covariance = kernel(X, X, length_scale=0.5, p=p)
    mean = np.zeros(len(X))
    samples = sample_gp(mean, covariance, num_samples=3)

    ax = axes[0, i]
    linestyle = '-' if add_lines else 'o'
    ax.plot(X, samples, linestyle, label='Sampled GP')
    ax.set_xlabel('$X$')
    ax.set_ylabel(r'$y \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$')
    ax.set_title(f'$\sigma = 0.25$, $p = {labels[i]}\pi$')

    ax = axes[1, i]
    ax.imshow(covariance, cmap='Purples', extent=(0, 1, 0, 1), origin='upper',
               aspect='auto', interpolation='nearest')
    ax.axis('equal')
    ax.axis('off')


plt.savefig('figures/gp_sampling_6.png')





