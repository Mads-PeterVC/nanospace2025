from agox.models.GPR import GPR

from bayes_nanospace2025.dataset import get_data
from agox.models.GPR.kernels import Constant as ConstantKernel, RBF, Noise

class ZeroDeltaFunc:

    def energy(self, *args, **kwargs):
        return 0

def make_model(descriptor=None, temp_atoms=None, length_scale=10, length_scale_bounds=(10, 100)):
    noise_level = 0.001
    kernel = ConstantKernel() * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + Noise(noise_level, noise_level_bounds=(noise_level, noise_level))    
    model = GPR(descriptor=descriptor, kernel=kernel, use_ray=False)

    return model

if __name__ == '__main__':

    model = make_model()

    from bayes_nanospace2025.interpolation import get_initial_training_data

    data, energies = get_initial_training_data()

    model.train_model(data, energies)
        