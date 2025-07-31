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


for length_scale in [0.01, 0.05, 0.1, 0.25, 0.5]:

    X = np.linspace(0, 1, 30)
    covariance = kernel(X, X, length_scale=length_scale)
    mean = np.zeros(len(X))
    samples = sample_gp(mean, covariance, num_samples=3)


    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    ax = axes[0]
    linestyle = '-o' if add_lines else 'o'
    ax.plot(X, samples, linestyle, label='Sampled GP')
    ax.set_xlabel('$X$')
    ax.set_ylabel(r'$y \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$')
    ax.set_title('GP Samples')

    # Plot the covariance matrix
    ax = axes[1]
    x, y = np.meshgrid(np.flip(X), np.flip(X))
    cax = ax.pcolormesh(x, y, covariance, cmap='Purples', edgecolor='k', linewidth=0.5)

    # Flip the y-axis to match the original orientation
    ax.set_ylim(ax.get_ylim()[::-1])
    # X-ticks and Y-ticks can only be in [0, 1] for this example
    ax.set_xticks(X)
    ax.set_yticks(X)
    ax.set_title(r'Covariance Matrix $\Sigma$')
    plt.colorbar(cax, ax=ax)
    ax.axis('equal')
    ax.axis('off')

    plt.savefig(f'gp_sampling_3_{length_scale}.png')





