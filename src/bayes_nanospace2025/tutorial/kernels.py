import numpy as np
from scipy.spatial.distance import cdist

class Kernel:
    def __init__(self):
        pass

    def __call__(self, x1, x2=None):
        """
        Call the kernel function with two sets of inputs.
        """
        raise NotImplementedError("Kernel must implement the __call__ method.")
    
    def __add__(self, other):
        """
        Add two kernels together.
        """
        if not isinstance(other, Kernel):
            raise TypeError("Can only add another Kernel instance.")
        return SumKernel(self, other)
    
    def __mul__(self, other):
        """
        Multiply two kernels together.
        """
        if not isinstance(other, Kernel):
            raise TypeError("Can only multiply by another Kernel instance.")
        return ProductKernel(self, other)
    
class SumKernel(Kernel):
    """
    Sum of two kernels.
    """

    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, x1, x2=None):
        return self.kernel1(x1, x2) + self.kernel2(x1, x2)
    
class ProductKernel(Kernel):
    """
    Product of two kernels.
    """

    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, x1, x2=None):
        return self.kernel1(x1, x2) * self.kernel2(x1, x2)


class RadialBasis(Kernel):
    """
    Radial Basis Function (RBF) kernel.
    """

    def __init__(self, length_scale=1.0):
        super().__init__()
        self.length_scale = length_scale

    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        dists = cdist(x1, x2, "sqeuclidean")  # Your code here.
        return np.exp(-dists / (2 * self.length_scale**2))  # Your code here.

class Constant(Kernel):
    """
    Constant kernel.
    """

    def __init__(self, amplitude=1.0):
        super().__init__()
        self.amplitude = amplitude

    def __call__(self, x1, x2=None):
        return self.amplitude 
    
class Noise(Kernel):
    """
    Noise kernel.
    """

    def __init__(self, noise_level=1e-3):
        super().__init__()
        self.noise_level = noise_level

    def __call__(self, x1, x2=None):
        if x2 is not None:
            return np.zeros((x1.shape[0], x2.shape[0]))
        else:
            return self.noise_level * np.eye(x1.shape[0])

class Periodic(Kernel):

    def __init__(self, length_scale=1.0, period=1.0):
        super().__init__()
        self.length_scale = length_scale
        self.period = period
    
    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        d = cdist(x1, x2, metric='euclidean')
        k = np.exp(-2 * np.sin(np.pi * d/self.period)**2 / self.length_scale**2)

        return k
    
class Linear(Kernel):

    def __init__(self, slope=1.0, intercept=0.0):
        super().__init__()
        self.slope = slope
        self.intercept = intercept
    
    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        return self.slope * np.dot(x1, x2.T) + self.intercept