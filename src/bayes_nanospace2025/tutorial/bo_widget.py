import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from bayes_nanospace2025.tutorial.bo_optimizer import OptimizationResult


class BOWidget:
    def add_mean(self, fig, result, X_plot, step):
        trace = go.Scatter(
            visible=step == 0,
            line=dict(color="rgba(26,150,65,1.0)", width=4),
            x=X_plot,
            y=result.predictions[step].mean,
            name=f"Iteration {step}",
        )
        fig.add_trace(trace, row=1, col=1)

    def add_observations(self, fig, result, X_plot, step):
        y = np.array(result.y_obs[0 : step + 1]).flatten()

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
            row=1,
            col=1,
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
            row=1,
            col=1,
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
            row=2,
            col=1,
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
            row=2,
            col=1,
        )

    def add_uncertainty(self, fig, result, X_plot, step):
        y_bottom = result.predictions[step].mean - np.sqrt(result.predictions[step].variance)
        y_top = result.predictions[step].mean + np.sqrt(result.predictions[step].variance)

        trace_bottom = go.Scatter(x=X_plot, y=y_bottom, fill=None, name=f"Iteration {step}", visible=step == 0, showlegend=False, line=dict(color="rgba(26,150,65,0.4)"))
        trace_top = go.Scatter(x=X_plot, y=y_top, fill="tonexty", name=f"Iteration {step}", visible=step == 0, fillcolor='rgba(26,150,65,0.3)', showlegend=False, line=dict(color="rgba(26,150,65,0.4)"))
        fig.add_trace(trace_bottom)
        fig.add_trace(trace_top)

    def visible_traces(self, step, fig):
        return [True if trace.name in [f"Iteration {step}", "truth"] else False for trace in fig.data]

    def plot(self, result: OptimizationResult, X_plot, truth=None):
        layout = go.Layout(
            width=900,
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
        )

        fig = go.Figure(layout=layout)

        # Create figure
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["GP Prediction", "Acquistion Function"],
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            figure=fig,
        )

        # The truth line
        fig.add_trace(
            go.Scatter(x=X_plot, y=truth, visible=True, name="truth", line=dict(color="mediumpurple", width=4))
        )

        if result is not None:
            iterations = len(result.y_obs)
            for step in range(iterations):
                self.add_mean(fig, result, X_plot, step)
                self.add_observations(fig, result, X_plot, step)
                self.add_acquisition_point(fig, result, X_plot, step)
                self.add_acquisition_function(fig, result, X_plot, step)
                self.add_uncertainty(fig, result, X_plot, step)

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


def plot_bo(
    result: OptimizationResult = None,
    X_plot=None,
    truth=None,
    X_obs=None,
    y_obs=None,
    predictions=None,
    acquisition_values=None,
    acquisition_indices=None,
):
    if result is None:
        acquisition_values = [acq_value.flatten() for acq_value in acquisition_values]
        acquisition_indices = [
            acq_index.item() if hasattr(acq_index, "item") else acq_index for acq_index in acquisition_indices
        ]
        X_obs = [x.item() for x in X_obs]
        y_obs = [y.item() for y in y_obs]

        result = OptimizationResult(
            X_obs=X_obs,
            y_obs=y_obs,
            predictions=predictions,
            acquisition_values=acquisition_values,
            acquisition_indices=acquisition_indices,
        )

    widget = BOWidget()
    widget.plot(result=result, X_plot=X_plot, truth=truth)
