'''
@Project: MPK-GNN
@File   : sample_learning.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    The implementation of Sample-based Learning Module
'''
import torch.nn as nn

from models.basic_layers.mlfpn_fc import MLFPN_FC


class SampleLearning(nn.Module):

    def __init__(self, num_omics, num_genes, slm_dim_1, slm_dim_2):
        super(SampleLearning, self).__init__()

        slm_dims = [num_omics * num_genes, slm_dim_1, slm_dim_2]
        self.network = MLFPN_FC(slm_dims, nn.ReLU())

    def forward(self, feat):
        return self.network(feat)
