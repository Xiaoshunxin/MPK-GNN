'''
@Project: MPK-GNN
@File   : ops_al.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Operations within algorithm
'''
import torch
import numpy as np
import torch.nn as nn
from typing import Optional, List
from collections import OrderedDict
from cytoolz.itertoolz import sliding_window

from models.basic_layers.graph_conv import GraphConvolution


def build_layer_units(layer_type: str, dims: List[int], act_func: Optional[nn.Module]) -> nn.Module:
    """
        Construct a multi-layer network accroding to layer type, dimensions and activation function
        Tips: the activation function is not used in the final layer
    :param layer_type: the type of each layer, such as linear or gcn
    :param dims: the list of dimensions
    :param act_func: the type of activation function
    :return:
    """
    # build the first n-1 layers
    layer_list = []
    for input_dim, output_dim in sliding_window(2, dims[:-1]):
        layer_list.append(single_unit(layer_type, input_dim, output_dim, act_func))

    # build the last layer
    layer_list.append(single_unit(layer_type, dims[-2], dims[-1], None))

    return nn.Sequential(*layer_list)


def single_unit(layer_type: str, input_dim: int, output_dim: int, act_func: Optional[nn.Module]):
    """
        Construct each layer
    :param layer_type: the type of current layer
    :param input_dim: the input dimension
    :param output_dim: the output dimension
    :param act_func: the activation function
    :return:
    """
    unit = []
    if layer_type == 'linear':
        unit.append(('linear', nn.Linear(input_dim, output_dim)))
    elif layer_type == 'gcn':
        unit.append(('gcn', GraphConvolution(input_dim, output_dim)))
    else:
        print("Please input correct layer type!")
        exit()

    if act_func is not None:
        unit.append(('act', act_func))

    return nn.Sequential(OrderedDict(unit))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m,  torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
