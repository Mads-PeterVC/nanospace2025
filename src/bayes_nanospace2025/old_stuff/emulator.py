from ase.calculators.emt import EMT 
from time import sleep

class Emulator:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sleep = 0
        self.count = 0

        self.cache = {}

    def calculate(self, atoms):

        energy = self.cache.get(atoms.info['value'], None)

        if energy is not None:
            return energy
        else:
            self.count += 1
            sleep(self.sleep)
            atoms.calc = EMT()
            energy = atoms.get_potential_energy()
            self.cache[atoms.info['value']] = energy
            return energy

    def reset(self):
        self.count = 0
        self.cache = {}

emulator = Emulator()

def get_true_energy(atoms, cheat=False):
    c = emulator.count
    E = emulator.calculate(atoms)
    if cheat:
        emulator.count = c
    return E

def calculation_count():
    return emulator.count

def reset_calculation_count():
    emulator.reset()