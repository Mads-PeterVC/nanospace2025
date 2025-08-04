from plotly import graph_objs as go
from bayes_nanospace2025.tutorial.bo_optimizer import OptimizationResult
import numpy as np
from plotly.subplots import make_subplots


class BOWidget2D:

    def add_prediction(self, fig, result, step):
        z = result.predictions[step].mean.reshape(self.z_shape)

        if self.plot_types == 'contour':
            trace = go.Contour(
                visible=step == 0,
                z=z,
                x0 = self.x0,
                dx = self.dx,
                y0 = self.y0,
                dy = self.dy,
                name=f"Iteration {step}",
                coloraxis='coloraxis',
                contours=dict(
                    start=0,
                    end=5,
                    size=0.5,
            ))
        else:
            trace = go.Surface(
                visible=step == 0,
                z=z,
                x=np.arange(self.x0, self.x0 + self.dx * self.z_shape[0], self.dx),
                y=np.arange(self.y0, self.y0 + self.dy * self.z_shape[1], self.dy),
                name=f"Iteration {step}",
                coloraxis='coloraxis')

        fig.add_trace(trace, row=1, col=1)

    def add_data_scatter(self, fig, result, step):
        x_obs_arr = np.array(result.X_obs).reshape(-1, 2)
        x = x_obs_arr[0:step, 0]
        y = x_obs_arr[0:step, 1]

        if self.plot_types == "contour":
            fig.add_trace(
                go.Scatter(
                    visible=step == 0,
                    mode="markers",
                    x=x,
                    y=y,
                    name=f"Iteration {step}",
                    marker=dict(color="red", size=8),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )        
        else:
            z = np.array(result.y_obs).flatten()[0:step]
            trace = go.Scatter3d(x=x, y=y, z=z, name=f"Iteration {step}", visible=False, 
                                 mode='markers')
            fig.add_trace(trace, row=1, col=1)


    def add_acquisition(self, fig, result, step):
        z = result.acquisition_values[step].reshape(self.z_shape)
        trace = go.Contour(
            visible=step == 0,
            z=z,
            x0 = self.x0,
            dx = self.dx,
            y0 = self.y0,
            dy = self.dy,
            name=f"Iteration {step}",
            coloraxis='coloraxis2',
            contours=dict(
                end=5,
                start=self.acq_min,
                size=0.5,
        ))
        fig.add_trace(trace, row=1, col=2)

    def add_acquisition_point(self, fig, result, step):
        acq_index = result.acquisition_indices[step]
        x = result.predictions[0].X_query[acq_index, 0]
        y = result.predictions[0].X_query[acq_index, 1]
        fig.add_trace(
            go.Scatter(
                visible=step == 0,
                mode="markers",
                x=[x],
                y=[y],
                name=f"Iteration {step}",
                marker=dict(color="green", size=10, symbol="cross"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    def visible_traces(self, step, fig):
        return [True if trace.name in [f"Iteration {step}", "truth"] else False for trace in fig.data]

    def plot(self, result: OptimizationResult, X1, X2, plot_types='contour'):
        layout = go.Layout(
        xaxis=dict(title="X1"),
        yaxis=dict(title="X2"),
        width=1000,
        height=500,
        margin = dict(l=50, r=50, t=50, b=50),
    )
        fig = go.Figure(layout=layout)
        fig = make_subplots(1, 2, figure=fig, subplot_titles=("GP Mean", "Acquisition Function"), 
                            specs=[[{"type": plot_types}, {"type": plot_types}]])

        self.dx = X1[0, 1] - X1[0, 0]
        self.dy = X2[1, 0] - X2[0, 0]
        self.x0 = np.min(X1)
        self.y0 = np.min(X2)
        self.z_shape = X1.shape
        self.acq_min = np.min(result.acquisition_values)
        self.plot_types = plot_types

        iterations = len(result.predictions)
        for step in range(iterations):
            self.add_prediction(fig, result, step)
            self.add_data_scatter(fig, result, step)
            self.add_acquisition(fig, result, step)
            self.add_acquisition_point(fig, result, step)

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


        fig.update_layout(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            showlegend=True,
        )

        fig.update_layout(
            coloraxis=dict(colorscale='deep_r', colorbar_x=0.47, colorbar_thickness=23),
            coloraxis2=dict(colorscale='matter_r', colorbar_x=1.0075, colorbar_thickness=23)
            )

        fig.update_layout(xaxis1=dict(range=[self.x0, X1.max()]),
                          yaxis1=dict(range=[self.y0, X2.max()]),
                          xaxis2=dict(range=[self.x0, X1.max()]),
                          yaxis2=dict(range=[self.y0, X2.max()]),
        )


        return fig

def plot_bo_2d(result: OptimizationResult, X1, X2, plot_types='contour'):
    """
    Plot the results of Bayesian optimization in 2D.

    Parameters:
    - result: OptimizationResult object containing the results of the optimization.
    - X1: 2D array for the first dimension.
    - X2: 2D array for the second dimension.
    - plot_types: Type of plot to use ('contour' or 'surface').

    Returns:
    - fig: Plotly figure object.
    """
    widget = BOWidget2D()
    return widget.plot(result, X1, X2, plot_types)