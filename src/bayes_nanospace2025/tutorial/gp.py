from dataclasses import dataclass
from typing import Callable
import numpy as np
from bayes_nanospace2025.tutorial.kernels import Kernel
from scipy.linalg import cho_factor, cho_solve


@dataclass
class PosteriorState:
    X_train: np.ndarray  # Training inputs
    y_train: np.ndarray  # Training outputs
    K_inv: np.ndarray  # Inverse of the covariance matrix for training inputs
    alpha: np.ndarray  # Vector for computing the posterior mean
    cholesky_factor: np.ndarray


@dataclass
class PredictionResult:
    X_query: np.ndarray  # Query points where predictions are made
    mean: np.ndarray  # Posterior mean at query points
    variance: np.ndarray  # Posterior variance at query points
    covariance: np.ndarray  # Posterior covariance matrix at query points


class GaussianProcess:
    def __init__(self, kernel: Kernel, prior_mean: Callable | float | None = None, jitter: float = 1e-9):
        self.kernel = kernel
        self.jitter = jitter
        self.prior_mean = prior_mean if prior_mean is not None else 0
        self.state = None  # This will hold the posterior state after conditioning on observed data

    def condition(self, X_obs, y_obs) -> None:
        """
        Condition the Gaussian Process on observed data.
        """
        K_obs = self.kernel(X_obs)  # Covariance matrix for the observations
        K_obs += self.jitter * np.eye(len(X_obs))  # Add jitter to the diagonal to ensure numerical stability

        # Use Cholesky decomposition to compute the inverse of the covariance matrix
        L, _ = cho_factor(K_obs, lower=True)  # Cholesky factorization
        K_obs_inv = cho_solve((L, True), np.eye(len(X_obs)))  # Inverse of the covariance matrix

        # If prior_mean is callable, evaluate it at X_obs; otherwise, use a constant prior mean
        prior_mean = self.prior_mean(X_obs) if callable(self.prior_mean) else np.full((len(X_obs), 1), self.prior_mean)
        y = y_obs - prior_mean  # Adjust observed outputs by subtracting the prior mean

        # Compute alpha for efficient posterior mean computation
        alpha = K_obs_inv @ y

        self.state = PosteriorState(X_train=X_obs.copy(), y_train=y.copy(), K_inv=K_obs_inv, alpha=alpha, cholesky_factor=L)

    def predict(self, X_query: np.ndarray) -> PredictionResult:
        """
        Predict the mean and variance at new query points.
        """
        state = self.state

        prior_mean = (
            self.prior_mean(X_query) if callable(self.prior_mean) else np.full((len(X_query), 1), self.prior_mean)
        )

        if state is None:
            K_qq = self.kernel(X_query)  # Prior covariance for the query points.
            return PredictionResult(
                X_query=X_query,
                mean=prior_mean.flatten(),
                variance=np.diag(K_qq),
                covariance=K_qq,
            )

        # Predicting the posterior mean
        K_query = self.kernel(X_query, state.X_train) # Your code here. 

        # K_query: (n_query, n_train) - alpha: (n_train, 1) - prior_mean: (n_query, 1)
        mu_posterior = K_query @ state.alpha + prior_mean  # Your code here.
        mu_posterior = mu_posterior.flatten()

        # Predicting the posterior covariance
        K_qq = self.kernel(X_query)  # Covariance matrix for the query points (n_query, n_query).
        cov_posterior = K_qq - K_query @ state.K_inv @ K_query.T  # Your code here. (n_query, n_query)

        # Compute the posterior variance as the diagonal of the covariance matrix
        var_posterior = np.diag(cov_posterior).copy() # (n_query, )
        var_posterior[var_posterior < 0] = 0  # Ensure non-negative variance

        return PredictionResult(X_query=X_query, mean=mu_posterior, variance=var_posterior, covariance=cov_posterior)

    def sample(self, X_query: np.ndarray, n_samples: int = 1, cov_jitter: float = 1e-6) -> np.ndarray:
        """
        Sample from the posterior distribution at query points.
        """
        prediction = self.predict(X_query)
        samples = np.random.multivariate_normal(
            prediction.mean.flatten(),
            prediction.covariance + np.eye(prediction.covariance.shape[0]) * cov_jitter,
            n_samples,
            check_valid="ignore",
        )
        return samples

    def log_marginal_likelihood(self) -> float:
        """
        Compute the log marginal likelihood of the observed data.
        """
        y = self.state.y_train
        K_inv = self.state.K_inv
        L = self.state.cholesky_factor

        lml = -0.5 * y.T @ K_inv @ y - np.sum(np.log(np.diag(L))) - len(y) / 2 * np.log(2 * np.pi)
        return lml.flatten()[0]
    
    def set_prior_mean(self, prior_mean: Callable | float) -> None:
        """
        Set the prior mean function.
        """
        self.prior_mean = prior_mean if prior_mean is not None else 0
        self.state = None