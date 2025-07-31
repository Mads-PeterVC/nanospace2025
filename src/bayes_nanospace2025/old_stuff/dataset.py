from ase import Atoms
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from agox.candidates import StandardCandidate

#from importlib_resources import files
#path = files('dataset')
import os
directory = os.path.dirname(__file__)
#print(directory)


def convert_to_candidate(data):
    candidates = []
    for atoms in data:
        candidate = StandardCandidate.from_atoms(Atoms(cell=atoms.get_cell()), atoms)

        # energy = candidate.get_potential_energy()
        # forces = candidate.get_forces()
        #candidate.positions -= candidate.get_center_of_mass()
        # calc = SinglePointCalculator(candidate, energy=energy, forces=forces)
        # candidate.calc = calc
        candidates.append(candidate)

    return candidates

def get_data():
    return convert_to_candidate(read(directory+'/data/neb_path.traj',index=':'))
