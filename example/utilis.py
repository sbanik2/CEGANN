import math

import numpy as np
import torch
from pymatgen import Lattice, Structure


def DistanceMatrix(latticeMatrix, frac_coordinates):
    '''
    Numpy fast calculation of distance matrix
    '''
    M = latticeMatrix
    a, b, c = (
        frac_coordinates[:, 0],
        frac_coordinates[:, 1],
        frac_coordinates[:, 2],
    )

    def getDist(mat):
        n, m = np.meshgrid(mat, mat)
        dist = m - n
        dist -= np.rint(dist)
        return dist

    da, db, dc = (
        getDist(a),
        getDist(b),
        getDist(c),
    )  # Fracrtional difference matrix

    # ---------cartesian differences------------

    DX = M[0][0] * da + M[1][0] * db + M[2][0] * dc
    DY = M[0][1] * da + M[1][1] * db + M[2][1] * dc
    DZ = M[0][2] * da + M[1][2] * db + M[2][2] * dc

    # -----------distance matrix--------------

    D = np.sqrt(np.square(DX) + np.square(DY) + np.square(DZ))

    return D, [DX, DY, DZ]


def StructFrmAtoms(atoms):
    lattice = Lattice(np.array(atoms['lattice_mat']))
    structure = Structure(
        lattice, atoms['elements'], atoms['coords'], to_unit_cell=True
    )
    return structure


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def get_factor(l):
    return math.sqrt((2 * l + 1) / 4 * math.pi)


def Concat(atom_feature, bond_feature, nbr_idx):

    N, M = nbr_idx.shape
    _, O = atom_feature.shape
    _, _, P = bond_feature.shape

    index = nbr_idx.unsqueeze(1).expand(N, M, M)
    xk = atom_feature[index, :]
    xj = atom_feature[nbr_idx, :]
    xi = atom_feature.unsqueeze(1).expand(N, M, O)
    xij = torch.cat([xi, xj], dim=2)
    xij = xij.unsqueeze(2).expand(N, M, M, 2 * O)
    xijk = torch.cat([xij, xk], dim=3)

    eij = bond_feature.unsqueeze(2).expand(N, M, M, P)
    eik = bond_feature[nbr_idx, :]
    eijk = torch.cat([eij, eik], dim=3)

    return torch.cat([xijk, eijk], dim=3)
