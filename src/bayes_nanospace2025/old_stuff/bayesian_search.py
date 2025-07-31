import numpy as np
from bayes_nanospace2025.interpolation import get_structure
from bayes_nanospace2025.emulator import get_true_energy
from bayes_nanospace2025.descriptors import ReactionCoordinateDescriptor
import ipywidgets as widgets
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
from agox.utils.plot import plot_atoms
from IPython.display import clear_output

from ase.calculators.singlepoint import SinglePointCalculator

class AcquisitionFunction:

    def __init__(self, mode='min', **kwargs):
        self.parameters = [key for key in kwargs.keys()]
        self.__dict__.update(kwargs)
        if mode == 'min':
            self.selector = np.argmin
        elif mode == 'max':
            self.selector = np.argmax

    def acquisition_function(self, E, sigma, structures, model):
        return None

    def update(self, key, value):
        self.__dict__[key] = value

    def get_parameter(self, key):
        return self.__dict__[key]

    def __call__(self, structures, model):
        E = np.array(model.predict_energy(structures))
        sigma = np.array(model.predict_uncertainty(structures))
        af = self.acquisition_function(E, sigma, structures, model)
        index = self.selector(af)
        return af, index

class LowerConfidenceBound(AcquisitionFunction):

    def acquisition_function(self, E, sigma, structures, model):
        return E - self.kappa * sigma

class LowerConfidenceBoundExponent(AcquisitionFunction):

    def acquisition_function(self, E, sigma, structures, model):
        return E - (self.kappa * sigma) ** self.exponent

class ExpectedImprovement(AcquisitionFunction):

    def __init__(self, **kwargs):
        super().__init__(mode='max', **kwargs)

    def acquisition_function(self, E, sigma, structures, model):

        mu = E
        mu_opt = np.min(mu)

        with np.errstate(divide='warn'):
            imp = mu_opt - mu - self.xi
            Z = imp / sigma 
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0 
        return ei 

class ProbabilityOfImprovement(AcquisitionFunction):

    def __init__(self, **kwargs):
        super().__init__(mode='max', **kwargs)

    def acquisition_function(self, E, sigma, structures, model):
        mu = E
        mu_opt = np.min(mu)

        # For maximization
        # mu = -E
        # mu_opt = np.max(mu)

        with np.errstate(divide='warn'):
            imp = mu_opt - mu - self.xi
            # For maximization.
            #imp = mu  - mu_opt - self.xi
            Z = imp / sigma 
            poi = norm.cdf(Z)
            poi[sigma == 0.0] = 0.0 
        return poi

class BayesianSearch:

    def __init__(self, model=None, acquisition=None):
        self.model = model
        self.acquisition_function = acquisition

        self._X_train = []
        self._y_train = []
        self._c_train = []

        self.structure_set = [get_structure(v) for v in np.linspace(0, 1, 201)]
        self.rc_descriptor = ReactionCoordinateDescriptor()
        self.c_set = np.array(self.rc_descriptor.get_features(self.structure_set))
        self.true_energies = np.array([get_true_energy(s, cheat=True) for s in self.structure_set])

        self.add_initial_training_data()

    def add_initial_training_data(self):
        structures = [get_structure(c) for c in [0.0, 0.05, 0.1]]
        E0 = [get_true_energy(s) for s in structures]
        self.add_training_example(structures, E0)

    def add_training_example(self, X_new, y_new):
        if type(X_new) == list:            
            self._X_train += X_new
            self._y_train += y_new
        else:
            self._X_train.append(X_new)
            self._y_train.append(y_new)

        self._c_train = np.array(self.rc_descriptor.get_features(self._X_train))

        self.train_model()

    def remove_training_example(self, index):
        del self.X_train[index]
        del self.y_train[index]

    def train_model(self):        

        for atoms, energy in zip(self._X_train, self._y_train):
            atoms.calc = SinglePointCalculator(atoms, energy=energy)

        self.model.train(self._X_train)

    def next(self):
        # Ask the acquisition function: 
        af, index = self.acquisition_function(self.structure_set, self.model)

        X_new = self.structure_set[index]
        y_new = get_true_energy(X_new)

        self.add_training_example(X_new, y_new)
    
    def back(self):
        self._X_train = self._X_train[0:-1]
        self._y_train = self._y_train[0:-1]
        self._c_train = self._c_train[0:-1]
        self.train_model()

    def reset(self):
        self._X_train = []
        self._y_train = []
        self._c_train = []
        self.add_initial_training_data()
        self.train_model()

    def get_prediction(self):
        E = np.array(self.model.predict_energy(self.structure_set))
        sigma = np.array(self.model.predict_uncertainty(self.structure_set))
        return E, sigma

    def plot(self, ax):        
        # Prediction:
        E, sigma = self.get_prediction()
        l_pred, = ax.plot(self.c_set, E, label='Prediction')
        l_uncer = ax.fill_between(self.c_set.flatten(), E-sigma, E+sigma, alpha=0.2, edgecolor='black', facecolor=l_pred.get_color())

        # Acquisition Function: 
        ax_acqui = ax.twinx()

        af, af_idx = self.acquisition_function(self.structure_set, self.model)
        l_acqui, = ax_acqui.plot(self.c_set, af, label='Acquisition Function', color='darkblue')        
        l_next, = ax_acqui.plot(self.c_set[af_idx], af[af_idx], color='red', marker='$x$', label='Next point', markersize=12)

        # Training
        l_train, = ax.plot(self._c_train, self._y_train, 'o', label='Train')

        # Truth
        l_true, = ax.plot(self.c_set, self.true_energies, '--', alpha=1, label='Truth')
        ax.set_ylabel('Energy [eV]')
        ax.set_xlabel('Path coordinate', fontsize=14)
        ax.set_xlim([0, 1])
        ax.legend(loc='upper left')

        title = ax.set_title(f'Calculation count: {len(self._X_train)}')

        ax_acqui.legend(loc='upper right')
        ax_acqui.set_ylabel('Acquisition function')

        l_true.set_alpha(0)

        return ax_acqui, dict(l_pred=l_pred, l_uncer=l_uncer, l_acqui=l_acqui, l_next=l_next, l_train=l_train, l_true=l_true, title=title)

def make_box_layout():
     return widgets.Layout(
        border='solid 1px black',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px'
     )
 
class BayesianSearchWidget(widgets.HBox):
     
    def __init__(self, bayes, descriptors=[], acquisition_functions=[]):
        super().__init__()

        self.bayes = bayes

        output = widgets.Output()

        with output:
            output.clear_output(wait=True)
            self.fig = plt.figure(constrained_layout=True, figsize=(10, 5))
            gs = GridSpec(3, 3, figure=self.fig)
            #self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
            self.ax = self.fig.add_subplot(gs[:, 0:2])
            self.structure_ax = self.fig.add_subplot(gs[:, 2])
            self.fig.show()

        self.ax_acqui, self.lines = self.bayes.plot(self.ax)

        self.ax_keys = ['l_pred', 'l_uncer', 'l_train', 'l_true']
        self.ax_acqui_keys = ['l_acqui', 'l_next']
         
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)
 
        # Define widgets:
        child_widgets = []

        # Structure selector:
        structure_slider = widgets.IntSlider(value=0, min=0, max=len(bayes.structure_set)-1, 
            description='Structure: ', disabled=False)
        structure_slider.observe(self.plot_structure, 'value')
        child_widgets.append(structure_slider)

        # Change descriptor: 
        names = [descriptor.__name__ for descriptor in descriptors]
        self.descriptors = {descriptor.__name__:descriptor for descriptor in descriptors}
        current_value = self.bayes.model.descriptor.__name__
        descriptor_selector = widgets.Dropdown(options=list(self.descriptors.keys()), 
                                                value=current_value,
                                                description='Descriptor: ')
        descriptor_selector.observe(self.update_descriptor, 'value')
        child_widgets.append(descriptor_selector)        


        # Acquisiton parameter sliders. 
        self.acquisition_functions = {acq.__class__.__name__:acq for acq in acquisition_functions}

        current_acq = self.bayes.acquisition_function.__class__.__name__
        selection_acquisition = widgets.Dropdown(options=list(self.acquisition_functions.keys()), 
                                                value=current_acq,
                                                description='Acquisition function: ')
        selection_acquisition.observe(self.update_acquisition_function, 'value')
        child_widgets.append(selection_acquisition)

        # Figure out all acquisition parameters.
        all_parameters = {}
        for acquisition_function in acquisition_functions:
            for parameter in acquisition_function.parameters:
                value = acquisition_function.get_parameter(parameter)
                if parameter in all_parameters.keys():
                    all_parameters[parameter].append(value)
                else:
                    all_parameters[parameter] = [value]

        active_parameters = bayes.acquisition_function.parameters

        self.acquisiton_widgets = {}
        for parameter, value in all_parameters.items(): #bayes.acquisition_function.parameters:
            if parameter in active_parameters:
                disabled = False
            else:
                disabled = True
            # float_slider = widgets.FloatSlider(value=bayes.acquisition_function.get_parameter(parameter),
            #     min=0, max=20, description=parameter, continuous_update=False)
            float_slider = widgets.FloatText(value=np.mean(value),
                description=parameter, disabled=disabled)

            float_slider.observe(self.update_acquisition_parameter, 'value')
            child_widgets.append(float_slider)
            self.acquisiton_widgets[parameter] = float_slider


        # Button to acquire next data point. 
        next_button = widgets.Button(description='Acquire', disabled=False, 
            tooltip='Acquire next data point', icon='chevrons-right', button_style='',
            layout=widgets.Layout(width='33%'))
        next_button.on_click(self.update_acquire_next)

        # Return button to delete previous data point. 
        return_button = widgets.Button(description='Return', disabled=False, 
            tooltip='Acquire next data point', icon='chevrons-left', button_style='',
            layout=widgets.Layout(width='33%'))
        return_button.on_click(self.toggle_return)

        # Reset button:
        reset_button = widgets.Button(description='Reset', disabled=False, 
            tooltip='Reset', icon='toilet', button_style='', layout=widgets.Layout(width='33%'))
        reset_button.on_click(self.toggle_reset)

        button_hbox = widgets.HBox([next_button, return_button, reset_button])
        child_widgets.append(button_hbox)

        # Toggle acquisiton function. 
        acquisiton_toggle = widgets.ToggleButton(description='Show acquisition function', 
            disabled=False, tooltip='Toggle show acqusition function', value=True)
        acquisiton_toggle.observe(self.toggle_acquisition, 'value')

        # Toggle true function. 
        true_toggle = widgets.ToggleButton(description='Show truth', 
            disabled=False, tooltip='Toggle show truth', value=False)
        true_toggle.observe(self.toggle_true_data, 'value')

        toggle_hbox = widgets.HBox([acquisiton_toggle, true_toggle])
        child_widgets.append(toggle_hbox)

        # Final setup. 
        controls = widgets.VBox(child_widgets)
        controls.layout = make_box_layout()
         
        out_box = widgets.HBox([controls])
        output.layout = make_box_layout()           
        # # add to children
        self.controls = controls
        self.children = [out_box, output]

        self.current_structure_index = 0
        self.plot_structure(None)
        self.update_ylimits()
        clear_output()
    
    def update_all(self):
        E, sigma = self.bayes.get_prediction()
        self.plot_prediction(E)
        self.plot_uncertianty(E, sigma)
        self.plot_training_points()
        self.plot_acquisition_function()
        self.plot_shown_energy(redraw=False)
        self.plot_title()
        self.update_ylimits()
        self.fig.canvas.draw()

    def update_acquisition_parameter(self, change):
        key = change.owner.description
        self.bayes.acquisition_function.update(key, change.new)
        self.plot_acquisition_function()
        self.update_ylimits()
        self.fig.canvas.draw()

    def update_acquisition_function(self, change):
        new_function = self.acquisition_functions[change.new]
        self.bayes.acquisition_function = new_function

        for key, widget in self.acquisiton_widgets.items():
            if key in new_function.parameters:
                widget.disabled = False
            else:
                widget.disabled = True

        self.plot_acquisition_function()
        self.update_ylimits()
        self.fig.canvas.draw()

    def update_descriptor(self, change):
        new_descriptor = self.descriptors[change.new]
        self.bayes.model.model.descriptor = new_descriptor
        self.bayes.train_model()
        self.update_all()

    def toggle_acquisition(self, change):
        line = self.lines['l_acqui']
        line.set_alpha(int(change.new))
        self.fig.canvas.draw()

    def toggle_true_data(self, change):
        line = self.lines['l_true']
        line.set_alpha(int(change.new))
        self.fig.canvas.draw()

    def toggle_reset(self, change):
        self.bayes.reset()
        self.update_all()

    def toggle_return(self, change):
        self.bayes.back()
        self.update_all()

    def update_acquire_next(self, change):
        self.bayes.next()
        self.update_all()

    def plot_prediction(self, E):
        line = self.lines['l_pred']
        line.set_ydata(E)

    def plot_uncertianty(self, E, sigma):
        self.ax.collections.clear()
        color = self.lines['l_pred'].get_color()
        l = self.ax.fill_between(self.bayes.c_set.flatten(), E-sigma, E+sigma, alpha=0.2, color=color, edgecolor='black')
        return np.min(E-sigma), np.max(E+sigma)

    def plot_title(self):
        title_obj = self.lines['title']
        title_obj.set_text(f'Calculation count: {len(self.bayes._X_train)}')

    def plot_training_points(self):
        line = self.lines['l_train']
        line.set_xdata(self.bayes._c_train)
        line.set_ydata(self.bayes._y_train)

    def plot_acquisition_function(self):
        line = self.lines['l_acqui']
        af, idx = self.bayes.acquisition_function(self.bayes.structure_set, self.bayes.model)
        line.set_ydata(af)
        line = self.lines['l_next']
        line.set_xdata(self.bayes.c_set[idx])        
        line.set_ydata(af[idx])

    def update_ylimits(self):
        self.update_ylimits_manual(self.ax_keys, self.ax)
        self.update_ylimits_manual(self.ax_acqui_keys, self.ax_acqui)

    def update_ylimits_manual(self, keys, ax):
        current_limits = [np.inf, -np.inf]
        for key in keys:
            line = self.lines[key]
            # if line.alpha == 0:
            #     continue
            try:
                ydata = line.get_ydata()
                lmin = np.min(ydata)
                lmax = np.max(ydata)
                current_limits[0] = np.min([lmin, current_limits[0]])
                current_limits[1] = np.max([lmax, current_limits[1]])
            except Exception as e:
                y_vertices = line.get_paths()[0].vertices[:, 1]
                lmin = y_vertices.min()
                lmax = y_vertices.max()
                current_limits[0] = np.min([lmin, current_limits[0]])
                current_limits[1] = np.max([lmax, current_limits[1]])

        for i, limit in enumerate(current_limits):
            if i == 0:
                if limit > 0:
                    factor = 0.94
                else:
                    factor = 1.06
            elif i == 1:
                if limit > 0:
                    factor = 1.06
                else:
                    factor = 0.94
            current_limits[i] = factor * limit

        ax.set_ylim(current_limits)

    def plot_structure(self, change):
        if change == None:
            index = 0
        else:
            index = change.new
            self.current_structure_index = index

        structure = self.bayes.structure_set[index]
        if change is None:        
            E = self.bayes.model.predict_energy(structure)
            l_shown, = self.ax.plot(self.bayes.c_set[index], E, 'gs', label='Shown', markersize=12, zorder=1)
            self.lines['l_shown'] = l_shown
            self.ax.legend(loc='upper left')
            self.structure_ax.set_xticks([])            
            self.structure_ax.set_yticks([])
        else:
            self.plot_shown_energy()

        self.structure_ax.clear()
        plot_atoms(ax=self.structure_ax, atoms=structure)    

    def plot_shown_energy(self, redraw=True):
        structure = self.bayes.structure_set[self.current_structure_index]
        E = self.bayes.model.predict_energy(structure)
        line = self.lines['l_shown']
        line.set_xdata(self.bayes.c_set[self.current_structure_index])
        line.set_ydata(E)
        if redraw:
            self.fig.canvas.draw()

if __name__ == '__main__':
    pass





        
        
        


        

        

