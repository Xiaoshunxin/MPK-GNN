'''
@Project: MPK-GNN
@File   : mpk.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    The implementation of the proposed framework
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sub_modules.projection import Projection
from models.sub_modules.task_specific import TaskSpecific
from models.sub_modules.sample_learning import SampleLearning
from models.sub_modules.feature_learning import FeatureLearning


class MPK(nn.Module):

    def __init__(self, num_omics, num_genes, num_classes, slm_dim_1, slm_dim_2, flm_gcn_dim_1,
                 flm_gcn_dim_2, pool_size, flm_fl_dim, pm_dim_1, pm_dim_2):
        super(MPK, self).__init__()

        # Define the sample-based learning module
        self.slm = SampleLearning(num_omics, num_genes, slm_dim_1, slm_dim_2)

        # Define the feature-based learning module
        self.flm = FeatureLearning(num_omics, num_genes, flm_gcn_dim_1, flm_gcn_dim_2, pool_size, flm_fl_dim)

        # Define the Projection module
        self.pm = Projection(flm_fl_dim, pm_dim_1, pm_dim_2)

        # Define the task-specific module
        self.tsm = TaskSpecific(slm_dim_2, flm_fl_dim, num_classes)

    def forward(self, x_in, adj_0, adj_1, adj_2):
        """
            x_in: the input feature matrix with shape [batch_size, num_genes, num_omics]
            L: the adjacency matrix with shape [num_genes, num_genes]
        """
        x = x_in  # the input of the feature-level learning module
        x_nn = x_in.view(x_in.size()[0], -1)  # Input of the SLM module with [batch_size, num_genes * num_omics]

        # Process of Sample-based learning module (slm)
        o_slm = self.slm(x_nn)  # [batch_size, slm_dim_2]

        # Process of Feature-based learning module (flm)
        o_flm_0 = self.flm(x, adj_0)  # [batch_size, flm_fl_dim]
        o_flm_1 = self.flm(x, adj_1)  # [batch_size, flm_fl_dim]
        o_flm_2 = self.flm(x, adj_2)  # [batch_size, flm_fl_dim]
        o_flm = torch.cat((o_flm_0, o_flm_1, o_flm_2), 1)  # [batch_size, flm_fl_dim * 3]

        # Process of the Projection module (pm)
        o_pm_0 = self.pm(o_flm_0)
        o_pm_1 = self.pm(o_flm_1)
        o_pm_2 = self.pm(o_flm_2)

        # Process of the Task-specific module (tsm)
        i_tsm = torch.cat((o_flm, o_slm), 1)
        o_tsm = self.tsm(i_tsm)

        return o_tsm, F.normalize(o_pm_0, dim=1), F.normalize(o_pm_1, dim=1), F.normalize(o_pm_2, dim=1)
