import numpy as np
from agox.models.descriptors import DescriptorBaseClass
from agox.models.descriptors import Fingerprint as FingerprintAGOX

class IndexDescriptor(DescriptorBaseClass):

    feature_types = ['global']

    def __init__(self, structures, **kwargs):
        super().__init__(use_cache=True, **kwargs)
        self.structures = structures

    def create_global_features(self, atoms):
        for index in range(len(self.structures)):
            if (atoms.positions == self.structures[index].positions).all():
                return np.array([index]) / len(self.structures)
        return 0

class ReactionCoordinateDescriptor(DescriptorBaseClass):

    descriptor_type = 'global'
    name = 'ReactionCoordinate'

    def __init__(self, factor=1, **kwargs):
        super().__init__(use_cache=True, environment=None, **kwargs)
        self.factor = factor

    def create_features(self, atoms):
        return np.array([atoms.info['value'] * self.factor])
    
    def get_number_of_centers(self):
        return 1    

class CartesianCoordinateDescriptor(DescriptorBaseClass):

    descriptor_type = 'global'
    name = 'CartesianCoordinate'

    def __init__(self, **kwargs):
        super().__init__(use_cache=True, environment=None, **kwargs)

    def create_features(self, atoms):
        return atoms.positions.flatten()
    
    def get_number_of_centers(self):
        return 1

class NearestDistanceDescriptor(DescriptorBaseClass):

    descriptor_type = 'global'
    name = 'NearestDistance'

    def __init__(self, **kwargs):
        super().__init__(environment=None, **kwargs)

    def create_features(self, atoms):
        distance_matrix = atoms.get_all_distances()
        distance_matrix += np.eye(len(atoms)) * 10
        return distance_matrix.min(axis=1)
    
    def get_number_of_centers(self):
        return 1

if __name__ == '__main__':

    from ase.io import read, write 

    data = read('../example_trajectory.traj', ':')
    for i, atoms in enumerate(data):
        atoms.info['value'] = i / len(data)

    descriptor = ReactionCoordinateDescriptor()

    Z = descriptor.get_global_features(data)

    print(Z)

