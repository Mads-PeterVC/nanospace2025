import numpy as np
from bayes_nanospace2025.tutorial.gp import PredictionResult, GaussianProcess
from typing import Callable

class AcquisitionFunction:
    def __init__(self):
        pass

    def compute(self, prediction: PredictionResult, GaussianProcess, X_query: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Returns the index of the next point to sample and the acquisition function values over the query points.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
class LowerConfidenceBound(AcquisitionFunction):
    def __init__(self, kappa: float = 2.0):
        super().__init__()
        self.kappa = kappa

    def compute(self, prediction: PredictionResult, GaussianProcess, X_query: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Compute the Lower Confidence Bound (LCB) acquisition function.
        """
        lcb = prediction.mean - self.kappa * np.sqrt(prediction.variance)  # YCH: Construct the LCB acquisition function
        next_index = np.argmin(lcb)  # YCH: Find the index of the minimum value in lcb
        return next_index, lcb
    
class ThompsonSampling(AcquisitionFunction):
    def __init__(self):
        super().__init__()

    def compute(self, prediction: PredictionResult, GaussianProcess, X_query: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Compute the Thompson sampling acquisition function.
        """
        sample = GaussianProcess.sample(X_query, n_samples=1).T  # YCH: Sample from the GP
        next_index = np.argmin(sample)  # YCH: Find the index of the minimum value in the sampled function
        return next_index, sample.flatten()
    
from dataclasses import dataclass, field


@dataclass
class OptimizationResult:
    X_obs: list = field(default_factory=list)  # List of observed input points
    y_obs: list = field(default_factory=list)  # List of observed output values
    predictions: list[np.ndarray] = field(default_factory=list)  # List of predictions made by the GP
    acquisition_values: list[np.ndarray] = field(default_factory=list)  # List of acquisition function values
    acquisition_indices: list[int] = field(default_factory=list)

    def update(
        self, X_new: np.ndarray, y_new: np.ndarray, prediction: np.ndarray, acq_values: np.ndarray, acq_index: int
    ):
        """
        Update the optimization result with new observations.
        """
        self.X_obs.append(X_new)
        self.y_obs.append(y_new)
        self.predictions.append(prediction)
        self.acquisition_values.append(acq_values)
        self.acquisition_indices.append(acq_index)

    def __repr__(self):
        string = ''
        string += f'X_obs = {len(self.X_obs)}\n'
        string += f'y_obs = {len(self.y_obs)}\n'
        string += f'acquisition_indices = {len(self.acquisition_indices)}\n'
        string += f'acquisition_values = {len(self.acquisition_values)} - {self.acquisition_values[0].shape}\n'
        return string



class BayesianOptimizer:
    def __init__(
        self,
        objective_function: Callable,
        gp: GaussianProcess,
        acquisition_function: AcquisitionFunction,
        objective_input_index: bool = False,
    ):
        self.objective_function = objective_function
        self.gp = gp
        self.acquisition_function = acquisition_function
        self.objective_input_index = objective_input_index

    def iterate(self, X_query):
        # Make predictions with the GP
        prediction = self.gp.predict(X_query)  # YCH: Make predictions with the GP

        # Compute the acquisition function
        acq_index, acq_values = self.acquisition_function.compute(
            prediction, self.gp, X_query
        )  # YCH: Compute the acquisition function

        # Find the next point to sample
        next_x, next_y = self.query_objective(X_query, acq_index)  # YCH: Query the objective function at the next point

        return next_x, next_y, prediction, acq_values, acq_index

    def query_objective(self, X_query, acq_index):
        next_x = X_query[acq_index]
        if not self.objective_input_index:
            next_y = self.objective_function(next_x.reshape(1, -1))
        else:
            next_y = self.objective_function(acq_index)
        return next_x, next_y

    def update_gp(self, results: OptimizationResult):
        """
        Update the Gaussian Process with new observations.
        """
        y_obs_arr = np.array(results.y_obs).reshape(-1, 1)  # Ensure y_obs is a 2D array
        X_obs_arr = np.atleast_2d(np.array(results.X_obs))

        self.gp.set_prior_mean(np.mean(y_obs_arr))
        self.gp.condition(X_obs_arr, y_obs_arr)

    def optimize(self, X_query: np.ndarray, n_iterations: int):
        results = OptimizationResult()

        for _ in range(n_iterations):
            next_x, next_y, prediction, acq_values, acq_index = self.iterate(X_query)  # YCH: Run an iteration of the BO
            results.update(
                next_x, next_y, prediction, acq_values, acq_index
            )  # YCH: Update the results with new observations - using results.update
            self.update_gp(results)  # YCH: Update the GP with new data

        return results