import numpy as np
import matplotlib.pyplot as plt

def plot_gp_prediction(ax = None, prediction = None, color="blue", label="Posterior Mean", X_plot=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if X_plot is None:
        X_plot = prediction.X_query

    ax.plot(X_plot, prediction.mean, color=color, label=label)
    ax.fill_between(
        X_plot.flatten(),
        prediction.mean.flatten() + 2 * np.sqrt(prediction.variance),
        prediction.mean.flatten() - 2 * np.sqrt(prediction.variance),
        color=color,
        alpha=0.2,
        label=f"{label} +/- 2 std",
    )