import numpy as np
from bayes_nanospace2025.dataset import get_data

class TrajectoryInterpolation:

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.length = len(self.trajectory)
    
    def convert(self, val):
        conv = val * self.length
        i0 = np.min([int(np.floor(conv)), self.length-1])
        i1 = np.min([int(np.ceil(conv)), self.length-1])
        d = conv - np.floor(conv)
        return i0, i1, d

    def get_structure(self, value):
        i0, i1, l = self.convert(value)

        P0 = self.trajectory[i0].get_positions()
        P1 = self.trajectory[i1].get_positions()

        atoms = self.trajectory[0].copy()
        atoms.positions = P0 + (P1 - P0) * l
        atoms.info['value'] = value
        return atoms

generator = TrajectoryInterpolation(get_data())

def get_structure(value):
    return generator.get_structure(value)

def get_initial_training_data():
    from ase.calculators.emt import EMT

    values = [0, 0.02, 0.04, 0.06, 0.08]
    structures = [get_structure(val) for val in values]
    energies = []
    for struc in structures:
        struc.calc = EMT()
        energies.append(struc.get_potential_energy())
    return structures, energies