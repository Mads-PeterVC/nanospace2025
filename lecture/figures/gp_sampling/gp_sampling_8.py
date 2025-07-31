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

def kernel(x1, x2, exponent=1.0, sigma_b=0.0, sigma_v=1.0, c=0.0):
    return sigma_b**2 + sigma_v**2*((x1[:, np.newaxis]-c) * (x2[np.newaxis, :]-c)) ** exponent


sz = 3
fig, axes = plt.subplots(2, 3, figsize=(3*sz, 2*sz), constrained_layout=True)

for i, v in enumerate([1, 2, 3]):

    X = np.linspace(0, 1, 100)
    covariance = kernel(X, X, c=0.5, sigma_v=1, exponent=v) #+ 1e-6 * np.eye(len(X))
    mean = np.zeros(len(X))
    samples = sample_gp(mean, covariance, num_samples=3)

    ax = axes[0, i]
    linestyle = '-' if add_lines else 'o'
    ax.plot(X, samples, linestyle, label='Sampled GP')
    ax.set_xlabel('$X$')
    ax.set_ylabel(r'$y \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$')
    ax.set_title(r'$\alpha$'+f'$ = {v:.1f}$, $c = 0.5$')

    ax = axes[1, i]
    ax.imshow(covariance, cmap='Purples', extent=(0, 1, 0, 1), origin='upper',
               aspect='auto', interpolation='nearest')

    ax.axis('equal')
    ax.axis('off')


plt.savefig('figures/gp_sampling_8.png')





