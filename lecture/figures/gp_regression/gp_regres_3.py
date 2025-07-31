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

print(K_mm_inv)

k_m = kernel(X, xm)
mean_post = k_m @ K_mm_inv @ ym
cov_post = kernel(X, X) - k_m @ K_mm_inv @ k_m.T

# Sample from posterior
samples_post = sample_gp(mean_post, cov_post, num_samples=100)

sz = 4
fig, ax = plt.subplots(2, 2, figsize=(2*sz, 2*sz), constrained_layout=True)

# Plot prior samples
ax[0, 0].plot(X, samples_prior, '-', label='Prior', color='gray', alpha=0.5, linewidth=1.0)
ax[0, 0].set_title('Prior Samples from GP')
ax[0, 0].set_xlabel('$X$')
ax[0, 0].set_ylabel(r'$y \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$')
ax[0, 0].set_xlim((0, 1))
ax[0, 0].set_ylim((-3, 3)) 

# Plot posterior samples
ax[0, 1].plot(xm, ym, 'o', label='Target Points', color='red', markersize=5, zorder=10)
ax[0, 1].plot(X, samples_post, '-', label='Posterior', color='gray', alpha=0.5, linewidth=1.0)
ax[0, 1].plot(X, mean_post, label='Posterior Mean', color='black', linewidth=2)
ax[0, 1].set_title('Posterior Samples from GP')
ax[0, 1].set_xlabel('$X$')
ax[0, 1].set_xlim((0, 1))
ax[0, 1].set_ylim((-3, 3))

# Plot covariance matrix
ax[1, 0].imshow(covariance, cmap='Purples', extent=(0, 1, 0, 1), origin='upper',
               aspect='auto', interpolation='nearest')

ax[1, 1].imshow(cov_post, cmap='Purples', extent=(0, 1, 0, 1), origin='upper',
               aspect='auto', interpolation='nearest')

ax[1, 0].axis('off')
ax[1, 1].axis('off')



plt.savefig('figures/gp_regres_3.png', dpi=300)








