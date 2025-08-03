import numpy as np
import matplotlib.pyplot as plt
from bayes_nanospace2025.lecture import set_plot_settings
from scipy.spatial.distance import cdist
from bayes_nanospace2025.tutorial.gp import GaussianProcess
from bayes_nanospace2025.tutorial.kernels import RadialBasis, Periodic, Constant, Noise, Linear

np.random.seed(42)  # For reproducibility

set_plot_settings()

plt.switch_backend('webagg')

# Posterior: 
xm = np.array([0.33, 0.66, 0.99]).reshape(-1, 1)
ym = np.array([0.2, 1.0, -1]).reshape(-1, 1)
X = np.linspace(0, 1, 1000).reshape(-1, 1)

kernels = [
    RadialBasis(length_scale=0.15),
    Periodic(length_scale=0.5, period=0.5),
    Constant(10)*Linear()*Linear()*Linear()*Linear() + Noise(noise_level=0.1),
]


sz = 4
fig, ax = plt.subplots(1, 3, figsize=(3*sz, 1*sz), constrained_layout=True)

for i, kernel in enumerate(kernels):

    gp = GaussianProcess(kernel=kernel)
    gp.condition(xm, ym)

    prediction = gp.predict(X)

    var = np.diag(prediction.covariance)
    std = np.sqrt(var)
    mean_post = prediction.mean

    # Plot posterior samples
    ax[i].plot(xm, ym, 'o', label='Target Points', color='red', markersize=5, zorder=10)
    ax[i].fill_between(X.flatten(), mean_post - 2*std, mean_post + 2*std, color='lightgray', alpha=1.0, label='Posterior Variance')
        
    ax[i].plot(X.flatten(), mean_post, label='Posterior Mean', color='black', linewidth=2)
    ax[i].set_xlabel('$X$')
    ax[i].set_xlim((0, 1))
    ax[i].set_ylim((-3, 3))


plt.savefig('figures/gp_kernels.png', dpi=300)