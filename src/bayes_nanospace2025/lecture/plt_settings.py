import matplotlib.pyplot as plt

def set_plot_settings():
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (8, 4),
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'grid.alpha': 0.5,
        'grid.linestyle': '--',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',        
    })

    # Use tex
    plt.rcParams['text.usetex'] = True

    # Set color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['mediumpurple', 'darkorange', 'forestgreen', 'crimson', 'goldenrod'])