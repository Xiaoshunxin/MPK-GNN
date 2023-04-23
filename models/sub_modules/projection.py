'''
@Project: MPK-GNN
@File   : projection.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    The implementation of the Projection Module
'''
import torch.nn as nn

from models.basic_layers.mlfpn_fc import MLFPN_FC


class Projection(nn.Module):
    
    def __init__(self, flm_fl_o, pm_dim_1, pm_dim_2):
        super(Projection, self).__init__()

        pm_dims = [flm_fl_o, pm_dim_1, pm_dim_2]
        self.network = MLFPN_FC(pm_dims, nn.ReLU())

    def forward(self, feat):
        return self.network(feat)

