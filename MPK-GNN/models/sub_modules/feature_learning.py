'''
@Project: MPK-GNN
@File   : feature_learning.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    The implementation of the Feature-based Learning Module.
    It contains three parts:
        a multi-layer graph convolution network
        a graph max pool layer
        a flatten layer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureLearning(nn.Module):

    def __init__(self, num_omics, num_genes, flm_gcn_dim_1, flm_gcn_dim_2, pool_size, flm_fl_dim):
        super(FeatureLearning, self).__init__()
        self.num_omics = num_omics
        self.num_genes = num_genes
        self.flm_gcn_dim_1 = flm_gcn_dim_1
        self.flm_gcn_dim_2 = flm_gcn_dim_2
        self.pool_size = pool_size

        # define the first layer of GCN
        self.gcn_1 = nn.Linear(self.num_omics, self.flm_gcn_dim_1)

        # define the second layer of GCN
        self.gcn_2 = nn.Linear(self.flm_gcn_dim_1, flm_gcn_dim_2)

        # define the flatten layer
        self.fl_dim_i = flm_gcn_dim_2 * (num_genes // pool_size)
        self.fl = nn.Linear(self.fl_dim_i, flm_fl_dim)

    def graph_conv_net(self, feat, adj):
        """
            feat: the input feature matrix with shape [batch_size, num_genes, num_omics]
            adj: the adjacency matrix with shape [num_genes, num_genes]
        """
        # Transform to the required input shape of the first layer of GCN
        batch_size, num_genes, num_omics = feat.size()
        x = feat.permute(1, 2, 0).contiguous()  # [num_genes, num_omics, batch_size]
        x = x.view([num_genes, num_omics * batch_size])  # [num_genes, num_omics * batch_size]

        # Learning process of the first layer of GCN
        x = torch.mm(adj, x)  # [num_genes, num_omics * batch_size]
        x = x.view([num_genes, num_omics, batch_size])  # [num_genes, num_omics, batch_size]
        x = x.permute(2, 0, 1).contiguous()  # [batch_size, num_genes, num_omics]
        x = x.view([batch_size * num_genes, num_omics])  # [batch_size * num_genes, num_omics]
        x = self.gcn_1(x)  # [batch_size * num_genes, flm_gcn_dim_1]
        x = F.relu(x)

        # Transform to the required input shape of the second layer of GCN
        x = x.view([batch_size, num_genes, self.flm_gcn_dim_1])  # [batch_size, num_genes, flm_gcn_dim_1]
        x = x.permute(1, 2, 0).contiguous()  # [num_genes, flm_gcn_dim_1, batch_size]
        x = x.view([num_genes, self.flm_gcn_dim_1 * batch_size])  # [num_genes, flm_gcn_dim_1 * batch_size]

        # Learning process of the second layer of GCN
        x = torch.mm(adj, x)  # [num_genes, flm_gcn_dim_1 * batch_size]
        x = x.view([num_genes, self.flm_gcn_dim_1, batch_size])  # [num_genes, flm_gcn_dim_1, batch_size]
        x = x.permute(2, 0, 1).contiguous()  # [batch_size, flm_gcn_dim_1, num_omics]
        x = x.view([batch_size * num_genes, self.flm_gcn_dim_1])  # [batch_size * num_genes, flm_gcn_dim_1]
        x = self.gcn_2(x)  # [batch_size * num_genes, flm_gcn_dim_2]
        x = F.relu(x)

        # The final output of the multi-layer GCN
        x = x.view([batch_size, num_genes, self.flm_gcn_dim_2])  # [batch_size, num_genes, flm_gcn_dim_2]

        return x

    def graph_max_pool(self, x):
        """
            x: the input feature matrix with shape [batch_size, num_genes, flm_gcn_dim_2]
        """
        if self.pool_size > 1:
            x = x.permute(0, 2, 1).contiguous()  # [batch_size, flm_gcn_dim_2, num_genes]
            x = nn.MaxPool1d(self.pool_size)(x)  # [batch_size, flm_gcn_dim_2, num_genes / self.pool_size]
            x = x.permute(0, 2, 1).contiguous()  # [batch_size, num_genes / self.pool_size, flm_gcn_dim_2]
            return x
        else:
            return x

    def forward(self, feat, adj):
        """
            :param feat: the input feature matrix with shape [batch_size, num_genes, num_omics]
            :param adj: the prior knowledge graphs, such as GGI or PPI, with shape [num_genes, num_genes]
        :return:
        """
        # Process of the multi-layer GCN
        x = self.graph_conv_net(feat, adj)  # [batch_size, num_genes, flm_gcn_dim_2]

        # Process of the Graph max pool layer
        x = self.graph_max_pool(x)  # [batch_size, num_genes / self.pool_size, flm_gcn_dim_2]

        # Process of the flatten layer
        x = x.view(-1, self.fl_dim_i)  # [batch_size, num_genes / pool_size * gcc_dim_o]
        x = self.fl(x)  # [batch_size, flm_fl_dim]
        x = F.relu(x)

        return x
