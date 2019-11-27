import numpy as np
import pandas as pd
import torch
import csv
from rdkit.Chem import MolFromSmiles, SDMolSupplier
from torch.utils.data import Dataset
import os
import scipy
from sklearn.utils import shuffle
import operator
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem

from .neural_fp import *

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor



def load_dataset(path, target_name, name):
    filename = target_name.lower() + '_' + name + '.csv'
    filepath = os.path.join(path, filename)
    x, y, target, _ = load_data_from_df(filepath, target_name)
    data_set = construct_dataset(x, y, target)
    data_set = MolDataset(data_set)
    return data_set


def load_data_from_df(dataset_path, target):    
    data_df = pd.read_csv(dataset_path)
    
    data_x = data_df.iloc[:, 0].values
    data_y = data_df.iloc[:, 1].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)
    
    x_all, y_all, target, mol_sizes = load_data_from_smiles(data_x, data_y, target)

    return (x_all, y_all, target, mol_sizes)


def load_data_from_smiles(smiles, labels, target, bondtype_freq =20, atomtype_freq =10):
    bondtype_dic = {}
    atomtype_dic = {}
    for smile in smiles:
        try:
            mol = MolFromSmiles(smile)
            bondtype_dic = fillBondType_dic(mol, bondtype_dic)
            atomtype_dic = fillAtomType_dic(mol, atomtype_dic)
        except AttributeError:
            pass
        else:
            pass

    sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

    filted_bondtype_list_order = []
    for i in range(0, len(bondtype_list_order)):
        if bondtype_list_number[i] > bondtype_freq:
            filted_bondtype_list_order.append(bondtype_list_order[i])
    filted_bondtype_list_order.append('Others')

    sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

    filted_atomtype_list_order = []
    for i in range(0, len(atomtype_list_order)):
        if atomtype_list_number[i] > atomtype_freq:
            filted_atomtype_list_order.append(atomtype_list_order[i])
    filted_atomtype_list_order.append('Others')

    print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order, filted_bondtype_list_order))

    # mol to graph
    i = 0
    mol_sizes = []
    x_all = []
    y_all = []
    print('Transfer mol to matrices')
    for mol_num, (smile, label) in enumerate(zip(smiles, labels)):
        try:
            mol = MolFromSmiles(smile)

            (afm, adj) = molToGraph(mol, filted_bondtype_list_order, filted_atomtype_list_order).dump_as_matrices_Att()
            x_all.append([afm, adj])
            y_all.append([label])
            mol_sizes.append(adj.shape[0])
        except AttributeError:
            print('the smile: {} has an error'.format(smile))
            print(mol_num)
        except RuntimeError:
            print('the smile: {} has an error'.format(smile))
            print(mol_num)
        except ValueError:
            print('the smile: {}, can not convert to graph structure'.format(smile))
            print(mol_num)
        else:
            pass

    # Normalize or not?
#     x_all = feature_normalize(x_all)

    print('Done.')
    return (x_all, y_all, target, mol_sizes)


def data_filter(x_all, y_all, target, sizes, tasks, size_cutoff=1000):
    idx_row = []
    for i in range(0, len(sizes)):
        if sizes[i] <= size_cutoff:
            idx_row.append(i)
    x_select = [x_all[i] for i in idx_row]
    y_select = [y_all[i] for i in idx_row]

    idx_col = []
    for task in tasks:
        for i in range(0, len(target)):
            if task == target[i]:
                idx_col.append(i)
    y_task = [[each_list[i] for i in idx_col] for each_list in y_select]

    return(x_select, y_task)


class MolDatum():
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """
    def __init__(self, x, label, target, index):
        self.adj = x[1]
        self.afm = x[0]
        self.label = label
        self.target = target
        self.index = index

        
def construct_dataset(x_all, y_all, target):
    output = []
    for i in range(len(x_all)):
        output.append(MolDatum(x_all[i], y_all[i], target, i))
    return(output)


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of MolDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        adj, afm  = self.data_list[key].adj, self.data_list[key].afm
        label = self.data_list[key].label
        return (adj, afm, label)
