'''
@Project: MPK-GNN
@File   : task_specific.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    The implementation of the Task-specific Module
'''
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecific(nn.Module):

    def __init__(self, slm_dim_2, flm_fl_dim, num_classes):
        super(TaskSpecific, self).__init__()
        self.network = nn.Linear(slm_dim_2 + flm_fl_dim * 3, num_classes)

    def forward(self, h):
        h = self.network(h)
        h = F.log_softmax(h)

        return h
