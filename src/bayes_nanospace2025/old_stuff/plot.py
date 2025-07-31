import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from ase.data.colors import jmol_colors
from ase.data import covalent_radii
from ase.io import read


def _plot_atoms(ax, atoms):

    jcols = [acol for acol in jmol_colors]

    acols = np.array([jcols[atom.number] for atom in atoms])

    ecol = [0,0,0]

    for ia in range(len(atoms)):
        acol = acols[ia]
        arad = covalent_radii[atoms[ia].number]*0.9

        pos = atoms[ia].position

        circ = Circle([pos[0],pos[1]],
                      fc=acol,
                      ec=ecol,
                      radius=arad,
                      lw=0.5,
                      zorder=1+pos[2]/1000)


        ax.add_patch(circ)

def plot_structures(strucs, limit=7, size_inch=1.4, plot_indices=False):
    N_strucs = len(strucs)
    if N_strucs == 0:
        print('Cannot plot - No structures given')
        return
    # if N_strucs > 20:
    #     print('Cannot plot more than 20 structure - please provide less')
        # return
    limits = [-limit, limit]

    ncols = np.min([5, N_strucs])
    nrows = (N_strucs + ncols - 1) // ncols
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows , figsize = (ncols * size_inch, nrows * size_inch))


    if N_strucs == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for idx in range(len(axes)):
        axes[idx].set_xlim(limits)
        axes[idx].set_ylim(limits)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    for idx,struc in enumerate(strucs):
        try:
            E = struc.get_potential_energy()
            E = r'E=${{{:8.3f}}}$'.format(E)
        except:
            E = 'NA'

        cell = struc.get_cell() * 1
        #struc = struc.repeat((3,3,1))
        #struc.translate(-cell[0]-cell[1])
        _plot_atoms(axes[idx], struc)
        axes[idx].set_aspect('equal')
        t = axes[idx].text(limits[0] + 0.5,limits[0] + 0.5, f'{idx}: {E}')

        if plot_indices:
            for i, atom in enumerate(struc):
                axes[idx].text(atom.x, atom.y, f'{i}', horizontalalignment='center', verticalalignment='center')


    fig.tight_layout()
    plt.subplots_adjust(hspace = 0, wspace = 0)
    return fig, axes


def feature_plot(atoms, feature):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(feature)), feature, width=.8)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Intensity')
    ax.set_xlim((0, len(feature)))

    axin = ax.inset_axes([0.70, 0.60, 0.40, 0.40])
    p = atoms.get_positions()
    limits = [np.min(p[:,:2])-2.,np.max(p[:,:2])+2.]
    axin.set_xlim(limits)
    axin.set_ylim(limits)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.axis('off')

    
    _plot_atoms(axin, atoms)

    axin.set_aspect('equal')

def projection_plot(X_projection, indices=None, colors=None):
    fig, ax = plt.subplots()
    ax.scatter(X_projection[:, 0], X_projection[:, 1], edgecolor='black', linewidth=0.5)

    if indices is not None:
        if colors is None:
            colors = 'orange'
        ax.scatter(X_projection[indices, 0], X_projection[indices, 1], edgecolor='black', linewidth=0.5, facecolor=colors, s=75)


    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    
def parity_subplot(ax,data,calculator,label=None):
    e1s = []
    e2s = []
    for struc in data:
        e1 = struc.get_potential_energy()
        e1s.append(e1)
        struc.set_calculator(calculator)
        e2 = struc.get_potential_energy()
        e2s.append(e2)
    ax.scatter(e1s,e2s,label=label)
    ax.plot([-42.5,-19.5],[-42.5,-19.5],'k')
    ax.axis('equal')

def path_energy_plot(model, training_data=None, ax=None):
    from bayes_nanospace2025.interpolation import get_structure
    from bayes_nanospace2025.emulator import get_true_energy
    from bayes_nanospace2025.descriptors import ReactionCoordinateDescriptor

    if ax is None:
        fig, ax = plt.subplots()

    # Prediction:
    structure_set = [get_structure(v) for v in np.linspace(0, 1, 201)]
    rc_descriptor = ReactionCoordinateDescriptor()
    c_set = np.array(rc_descriptor.get_features(structure_set)).reshape(-1, 1)


    E = np.array([model.predict_energy(s) for s in structure_set])
    sigma = np.array([model.predict_uncertainty(s) for s in structure_set])

    ax.plot(c_set, E, label='Prediction')



    ax.fill_between(c_set.flatten(), E-sigma, E+sigma, alpha=0.2, edgecolor='black')

    if training_data is not None:
        y_train = [get_true_energy(s, cheat=True) for s in training_data]
        c_train = rc_descriptor.get_features(training_data)
        ax.plot(c_train, y_train, 'o', label='Train')

    ax.set_ylabel('Energy [eV]', fontsize=14)
    ax.set_xlabel('Path coordinate', fontsize=14)
    ax.set_xlim([0, 1])

    ax.legend()
    return fig, ax

def prob_of_imp_figure():
    fig, ax = plt.subplots() 

    x_pred = np.linspace(0, 5) 
    ymin = 0.5
    xmin = 2
    y_pred = (x_pred-xmin)**2 + ymin

    ax.plot(x_pred, y_pred)

    y = np.linspace(-3, 8, 100)
    x0 = 2.75
    y0 = (x0-2)**2 + ymin
    x = np.exp(-(y-y0)**2) + x0

    l2, = ax.plot(x, y)
    ax.plot([x0, x.max()], [y0, y0], 'k--', alpha=0.5)

    ax.plot(xmin, ymin, 'o', label="f(x')")

    #ax.plot([0, 5], [ymin, ymin], 'k--', alpha=0.25)

    y_fill = np.linspace(y.min(), ymin)
    x1_fill = np.ones_like(y_fill) * x0
    x2_fill = np.exp(-(y_fill-y0)**2) + x0
    ax.fill_betweenx(y_fill, x1_fill, x2_fill, facecolor=l2.get_color(), edgecolor='black', alpha=0.25)

    ax.plot([xmin, xmin], [y.min(), ymin], 'k--', alpha=0.5)

    x_cut = x[np.argmin(y-ymin)]
    ax.plot([xmin, x_cut], [ymin, ymin], 'k--', alpha=0.5)

    ax.legend(fontsize=14)
    ax.set_xticks([2], ["x'"])
    ax.set_yticks([])
    ax.set_xlim([0, 5])
    ax.set_ylim([y.min(), y.max()])
    return fig, ax 
#display(fig); plt.close(fig)