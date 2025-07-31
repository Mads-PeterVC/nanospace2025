from bayes_nanospace2025.tutorial.kernels import RadialBasis, Noise, Constant
from bayes_nanospace2025.tutorial.gp import GaussianProcess
from bayes_nanospace2025.tutorial.plot_gp import plot_gp_prediction

class YourCodeHereError(NotImplementedError):
    def __init__(self, message="You need to implement this part of the code!"):
        super().__init__(message)

def YourCodeHere(message="You need to implement this part"):
    raise YourCodeHereError(message)
