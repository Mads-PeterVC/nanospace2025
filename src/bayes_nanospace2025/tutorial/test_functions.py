import numpy as np

def rastrigin(x):
    """
    Rastrigin function, a common test function for optimization algorithms.
    """
    n = x.shape[1]
    A = 10
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

def himmelblau(x: np.ndarray) -> np.ndarray:
    """
    Himmelblau's function, a common test function for optimization algorithms.
    """

    n = x.shape[1]

    if n != 2:
        y = 2
        x = x.flatten()
    else:
        x, y = x[:, 0], x[:, 1]

    return ((x**2 + y - 11)**2 + (x + y**2 - 7)**2) / 100


    
