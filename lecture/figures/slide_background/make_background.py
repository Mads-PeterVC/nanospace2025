from bayes_nanospace2025 import GaussianProcess, RadialBasis, Noise, Constant
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([-3, -2, 0, 1, 4]).reshape(-1, 1)
y = np.array([-1, 0, 1, 0, -1]).reshape(-1, 1)

# Set up GP
kernel = Constant() * RadialBasis(length_scale=1.0) + Noise(noise_level=1e-6)
gp = GaussianProcess(kernel=kernel)

gp.condition(X_obs=X, y_obs=y)

# Query
X_query = np.linspace(-5, 5, 1000).reshape(-1, 1)
samples = gp.sample(X_query=X_query, n_samples=100)

prediction = gp.predict(X_query=X_query)

# Slide background figure
# 16x9 aspect ratio
fig, ax = plt.subplots(figsize=(2*16, 2*9), layout='constrained')
ax.plot(X_query, samples.T, color="gray", linewidth=0.25, alpha=1.0)
ax.plot(X_query, prediction.mean, color="mediumpurple", linewidth=2)
ax.fill_between(X_query.flatten(),
                prediction.mean - 2*np.sqrt(prediction.variance),
                prediction.mean + 2*np.sqrt(prediction.variance),
                color="mediumpurple", alpha=0.25, label='Uncertainty', edgecolor='mediumpurple')
ax.scatter(X, y, color='red', label='Observations', zorder=5)

ax.set_xlim(-5, 5)
ax.axis('off')


plt.savefig('slide_background.png', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig('slide_background.pdf', dpi=600, bbox_inches='tight', transparent=True)



