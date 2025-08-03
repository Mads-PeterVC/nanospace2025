import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class BOWidget:
    def add_mean(self, fig, result, X_plot, step):
        trace = go.Scatter(
            visible=step == 0,
            line=dict(color="#00CED1", width=4),
            x=X_plot,
            y=result.predictions[step].mean,
            name=f"Iteration {step}",
        )
        fig.add_trace(trace, row=1, col=1)

    def add_observations(self, fig, result, X_plot, step):
        y = np.array(result.y_obs[0:step + 1]).flatten()

        fig.add_trace(
            go.Scatter(
                visible=step == 0,
                mode="markers",
                x=X_plot[result.acquisition_indices[0:step]],
                y=y,
                name=f"Iteration {step}",
                marker=dict(color="red", size=8),
                showlegend=False,
            ),
            row=1, col=1
        )

    def add_acquisition_point(self, fig, result, X_plot, step):
        y = np.array(result.y_obs[step : step + 1]).flatten()
        fig.add_trace(
            go.Scatter(
                visible=step == 0,
                mode="markers",
                x=X_plot[result.acquisition_indices[step : step + 1]],
                y=y,
                name=f"Iteration {step}",
                marker=dict(color="green", size=16),
                showlegend=False,
            ),
            row=1, col=1
        )

    def add_acquisition_function(self, fig, result, X_plot, step):
        fig.add_trace(
            go.Scatter(
                visible=step == 0,
                line=dict(color="orange", width=4),
                x=X_plot,
                y=result.acquisition_values[step],
                name=f"Iteration {step}",
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                visible=step == 0,
                mode="markers",
                x=X_plot[result.acquisition_indices[step : step + 1]],
                y=[result.acquisition_values[step][result.acquisition_indices[step]]],
                name=f"Iteration {step}",
                marker=dict(color="green", size=16),
                showlegend=False,
            ),
            row=2, col=1)

    def visible_traces(self, step, fig):
        return [True if trace.name in [f"Iteration {step}", "truth"] else False for trace in fig.data]

    def plot(self, result, X_plot, truth=None):
        # Create figure
        fig = make_subplots(
            rows=2, cols=1, subplot_titles=["GP Prediction", "Acquistion Function"], shared_xaxes=True, 
            vertical_spacing=0.1, horizontal_spacing=0.05
        )

        # Add traces, one for each slider step
        iterations = len(result.y_obs)

        # The truth line
        fig.add_trace(
            go.Scatter(x=X_plot, y=truth, visible=True, name="truth", line=dict(color="mediumpurple", width=4))
        )

        for step in range(iterations):
            self.add_mean(fig, result, X_plot, step)
            self.add_observations(fig, result, X_plot, step)
            self.add_acquisition_point(fig, result, X_plot, step)
            self.add_acquisition_function(fig, result, X_plot, step)


        # Create and add slider
        steps = []
        for i in range(iterations):
            step = dict(
                method="update",
                args=[{"visible": self.visible_traces(i, fig)}],  # layout attribute
            )
            steps.append(step)

        sliders = [dict(active=0, currentvalue={}, pad={"t": 25}, steps=steps)]
        fig.update_layout(sliders=sliders)
        fig.show()


def plot_bayes_opt_plotly(result=None, X_plot=None, truth=None):
    widget = BOWidget()
    widget.plot(result, X_plot, truth)
