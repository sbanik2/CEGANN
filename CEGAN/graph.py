from time import time

import numpy as np
import torch
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset

from utilis import convert

torch.set_default_dtype(torch.float32)

# =======================================


class Graph(object):
    '''
    Graph object fro creation of atomic graphs with bond and node attributes from pymatgen structure
    '''

    def __init__(
        self,
        neighbors=12,
        rcut=0,
        delta=1,
    ):

        self.neighbors = neighbors
        self.rcut = rcut
        self.delta = delta
        # self.atom = []
        self.bond = []
        self.nbr = []
        self.angle_cosines = []

    def setGraphFea(self, structure):

        if self.rcut > 0:
            pass
        else:
            species = [site.specie.symbol for site in structure.sites]
            self.rcut = max(
                [Element(elm).atomic_radius * 3 for elm in species]
            )

        all_nbrs = structure.get_all_neighbors(self.rcut, include_index=True)

        len_nbrs = np.array([len(nbr) for nbr in all_nbrs])

        indexes = np.where((len_nbrs < self.neighbors))[0]

        # print(self.rcut,len(indexes))

        for i in indexes:
            cut = self.rcut
            curr_N = len(all_nbrs[i])
            while curr_N < self.neighbors:
                cut += self.delta
                #  print("I am here")
                nbr = structure.get_neighbors(structure[i], cut)
                # print("I am here1")
                curr_N = len(nbr)
            # print("got it",count)
            # count +=1
            all_nbrs[i] = nbr

        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        self.nbr = torch.LongTensor(
            [
                list(map(lambda x: x[2], nbrs[: self.neighbors]))
                for nbrs in all_nbrs
            ]
        )
        self.bond = torch.Tensor(
            [
                list(map(lambda x: x[1], nbrs[: self.neighbors]))
                for nbrs in all_nbrs
            ]
        )

        cart_coords = torch.Tensor(np.array(
            [structure[i].coords for i in range(len(structure))]
        ))
        atom_nbr_fea = torch.Tensor(np.array(
            [
                list(map(lambda x: x[0].coords, nbrs[: self.neighbors]))
                for nbrs in all_nbrs
            ]
        ))
        centre_coords = cart_coords.unsqueeze(1).expand(
            len(structure), self.neighbors, 3
        )
        dxyz = atom_nbr_fea - centre_coords
        r = self.bond.unsqueeze(2)
        self.angle_cosines = torch.matmul(
            dxyz, torch.swapaxes(dxyz, 1, 2)
        ) / torch.matmul(r, torch.swapaxes(r, 1, 2))


# -------------------------------------------------


class load_graphs_targets(object):

    '''
    structureData should
    be in dict format
                      structure:{pymatgen structure},
                      property:{}
                      formula: None or formula
    if not from database
    '''

    def __init__(self, neighbors=12, rcut=0, delta=1):

        self.neighbors = neighbors
        self.rcut = rcut
        self.delta = delta

    def load(self, data):
        structure = data["structure"]
        target = data["target"]
        # print(target)
        graph = Graph(
            neighbors=self.neighbors, rcut=self.rcut, delta=self.delta
        )
        # try:
        # print("graphs")
        graph.setGraphFea(structure)
        # print("graphs done")
        return (graph, target)
        # except:
        #    return None


def process(func, tasks, n_proc, mp_load=False, mp_pool=None):

    if mp_load:
        results = []
        chunks = [tasks[i : i + n_proc] for i in range(0, len(tasks), n_proc)]
        for chunk in chunks:
            # print("chunks")
            r = mp_pool.map_async(func, chunk, callback=results.append)
            r.wait()
        mp_pool.close()
        mp_pool.join()
        return results[0]
    else:
        # print("chunks")
        return [func(task) for task in tasks]


# --------------------------------------------------------------


class CrystalGraphDataset(Dataset):
    '''
    A Crystal graph dataset container for genrating and loading pytorch dataset to be passed to train test and validation loader
    '''

    def __init__(
        self,
        dataset,
        neighbors=12,
        rcut=0,
        delta=1,
        mp_load=False,
        mp_pool=None,
        mp_cpu_count=None,
        **kwargs
    ):

        # ================================
        print("Loading {} graphs .......".format(len(dataset)))
        # =================================

        t1 = time()

        load_graphs = load_graphs_targets(
            neighbors=neighbors,
            rcut=rcut,
            delta=delta,
        )

        results = process(
            load_graphs.load,
            dataset,
            mp_cpu_count,
            mp_load=mp_load,
            mp_pool=mp_pool,
        )

        # print(results)

        self.graphs = [res[0] for res in results if res is not None]

        self.targets = [
            torch.LongTensor(res[1]) for res in results if res is not None
        ]
        # print(self.targets)
        self.binarizer = LabelBinarizer()
        self.binarizer.fit(torch.cat(self.targets))

        t2 = time()
        print("Total time taken {}".format(convert(t2 - t1)))

        self.size = len(self.targets)

    def collate(self, datalist):

        bond_feature, nbr_idx, angular_feature, crys_idx, targets = (
            [],
            [],
            [],
            [],
            [],
        )

        index = 0

        for (bond_fea, idx, angular_fea), targ in datalist:
            Natoms = bond_fea.shape[0]

            bond_feature.append(bond_fea)
            angular_feature.append(angular_fea)

            nbr_idx.append(idx + index)
            crys_idx.append([index, index + Natoms])
            targets.append(targ)
            index += Natoms

        return (
            torch.cat(bond_feature, dim=0),
            torch.cat(angular_feature, dim=0),
            torch.cat(nbr_idx, dim=0),
            torch.LongTensor(crys_idx),
            torch.cat(targets, dim=0),
        )

    def __getitem__(self, idx):

        graph = self.graphs[idx]
        bond_feature = graph.bond
        nbr_idx = graph.nbr
        angular_feature = graph.angle_cosines
        target = self.targets[idx]

        return (bond_feature, nbr_idx, angular_feature), target


# --------------------------------------------


def prepare_batch_fn(batch, device, non_blocking):

    # print(device,non_blocking)

    (bond_feature, angular_feature, nbr_idx, crys_idx, target) = batch

    # print(crys_idx)

    return (
        bond_feature.to(device, non_blocking=non_blocking),
        angular_feature.to(device, non_blocking=non_blocking),
        nbr_idx.to(device, non_blocking=non_blocking),
        crys_idx.to(device, non_blocking=non_blocking),
    ), target.to(device, non_blocking=non_blocking)


# ----------------------------
