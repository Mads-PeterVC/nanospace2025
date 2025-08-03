from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from importlib.resources import files
import numpy as np

def get_atoms_data():
    """    
    """
    path = files('bayes_nanospace2025.tutorial').joinpath('data/neb_path.traj')
    data = read(path, index=':')
    for atoms in data:
        atoms.calc = None

    return data


if __name__ == "__main__":
    # Example usage
    atoms_data = get_atoms_data()



