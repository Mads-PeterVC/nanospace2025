import numpy as np
import matplotlib.pyplot as plt

def plot_gp_prediction(ax = None, prediction = None, color="blue", label="Posterior Mean"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(prediction.X_query, prediction.mean, color=color, label=label)
    ax.fill_between(
        prediction.X_query.flatten(),
        prediction.mean.flatten() + 2 * np.sqrt(prediction.variance),
        prediction.mean.flatten() - 2 * np.sqrt(prediction.variance),
        color=color,
        alpha=0.2,
        label=f"{label} +/- 2 std",
    )