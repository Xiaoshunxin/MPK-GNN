'''
@Project: MOA-GNN
@File   : ops_io.py
@Time   : 2022-04-05 15:24 
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    
'''
import os
import time
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import preprocessing

from utils.ops_al import sparse_mx_to_torch_sparse_tensor


def process_data(num_genes, ratio):
    print('Process data...')
    t_start = time.time()

    expression_data_path = '/home/xiaosx/data/moa_data/TCGA/original_data/common_expression_data.tsv'
    cnv_data_path = '/home/xiaosx/data/moa_data/TCGA/original_data/common_cnv_data.tsv'
    expression_variance_file = '/home/xiaosx/data/moa_data/TCGA/original_data/expression_variance.tsv'
    shuffle_index_path = '/home/xiaosx/data/moa_data/TCGA/original_data/common_shuffle_index.tsv'
    adj_matrix_file_1 = '/home/xiaosx/data/moa_data/TCGA/original_data/adj_matrix_biogrid.npz'
    adj_matrix_file_2 = '/home/xiaosx/data/moa_data/TCGA/original_data/adj_matrix_coexpression.npz'
    adj_matrix_file_3 = '/home/xiaosx/data/moa_data/TCGA/original_data/adj_matrix_string.npz'
    non_null_index_path = '/home/xiaosx/data/moa_data/TCGA/original_data/biogrid_non_null.csv'

    expr_all_data, cnv_all_data = load_multiomics_data(expression_data_path, cnv_data_path)

    adj_1, adj_2, adj_3, train_data_all, labels, shuffle_index = downSampling_multiomics_data(
        expression_variance_path=expression_variance_file,
        expression_data=expr_all_data,
        cnv_data=cnv_all_data,
        non_null_index_path=non_null_index_path,
        shuffle_index_path=shuffle_index_path,
        adj_matrix_file_1=adj_matrix_file_1,
        adj_matrix_file_2=adj_matrix_file_2,
        adj_matrix_file_3=adj_matrix_file_3,
        number_gene=num_genes,
        singleton=False)

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)

    # Generate train and test part
    shuffle_index = shuffle_index.astype(np.int32).reshape(-1)

    train_size, val_size = int(len(shuffle_index) * ratio), int(len(shuffle_index) * (ratio+0.1))
    train_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[0:train_size]]
    val_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[train_size:val_size]]
    test_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[val_size:]]
    train_labels = labels[np.array(shuffle_index[0:train_size])].astype(np.int64)
    val_labels = labels[shuffle_index[train_size:val_size]].astype(np.int64)
    test_labels = labels[shuffle_index[val_size:]].astype(np.int64)

    # ll, cnt = np.unique(train_labels, return_counts=True)
    num_classes = len(np.unique(labels))

    # construct the normalized adjacency matrix of all prior networks
    L_1 = construct_adjacency_hat(adj_1)
    L_2 = construct_adjacency_hat(adj_2)
    L_3 = construct_adjacency_hat(adj_3)

    train_data = torch.FloatTensor(train_data)  # shape: num_samples * num_genes * num_omics
    train_labels = torch.LongTensor(train_labels)  # shape: num_samples
    val_data = torch.FloatTensor(val_data)
    val_labels = torch.LongTensor(val_labels)
    test_data = torch.FloatTensor(test_data)
    test_labels = torch.LongTensor(test_labels)

    print('Process done: ', time.time() - t_start)

    # save the processed data
    save_path = 'data/processed_data/' + str(num_genes) + '_' + str(ratio) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(train_data, save_path + 'train_data.pt')
    torch.save(train_labels, save_path + 'train_labels.pt')
    torch.save(val_data, save_path + 'val_data.pt')
    torch.save(val_labels, save_path + 'val_labels.pt')
    torch.save(test_data, save_path + 'test_data.pt')
    torch.save(test_labels, save_path + 'test_labels.pt')
    sp.save_npz(save_path + 'L_1.npz', L_1)
    sp.save_npz(save_path + 'L_2.npz', L_2)
    sp.save_npz(save_path + 'L_3.npz', L_3)
    torch.save(num_classes, save_path + 'num_classes.pt')

    L_1 = sparse_mx_to_torch_sparse_tensor(L_1)
    L_2 = sparse_mx_to_torch_sparse_tensor(L_2)
    L_3 = sparse_mx_to_torch_sparse_tensor(L_3)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, L_1, L_2, L_3, num_classes


def load_processed_data(num_genes, ratio):
    save_path = 'data/processed_data/' + str(num_genes) + '_' + str(ratio) + '/'
    train_data = torch.load(save_path + 'train_data.pt')
    train_labels = torch.load(save_path + 'train_labels.pt')
    val_data = torch.load(save_path + 'val_data.pt')
    val_labels = torch.load(save_path + 'val_labels.pt')
    test_data = torch.load(save_path + 'test_data.pt')
    test_labels = torch.load(save_path + 'test_labels.pt')
    num_classes = torch.load(save_path + 'num_classes.pt')
    L_1 = sp.load_npz(save_path + 'L_1.npz')
    L_1 = sparse_mx_to_torch_sparse_tensor(L_1)
    L_2 = sp.load_npz(save_path + 'L_2.npz')
    L_2 = sparse_mx_to_torch_sparse_tensor(L_2)
    L_3 = sp.load_npz(save_path + 'L_3.npz')
    L_3 = sparse_mx_to_torch_sparse_tensor(L_3)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, L_1, L_2, L_3, num_classes


def load_multiomics_data(expression_data_path, cnv_data_path):
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    cnv_data = pd.read_csv(cnv_data_path, sep='\t', index_col=0, header=0)
    return expression_data, cnv_data


def downSampling_multiomics_data(expression_variance_path, expression_data, cnv_data, shuffle_index_path,
                                 adj_matrix_file_1, non_null_index_path, adj_matrix_file_2, adj_matrix_file_3,
                                 number_gene, singleton=False):
    # obtain high varaince gene list
    high_variance_gene_list, high_variance_gene_index = \
        high_variance_expression_gene(expression_variance_path, non_null_index_path, number_gene, singleton)

    # get labels before filtering columns
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1

    # filter multi-omics data by gene list
    expression_data = expression_data.loc[:, high_variance_gene_list]
    cnv_data = cnv_data.loc[:, high_variance_gene_list]
    num_samples = cnv_data.shape[0]

    # concatenate expr and cnv
    data = np.asarray(cnv_data).reshape(num_samples, -1, 1)
    data = np.concatenate([data, np.asarray(expression_data).reshape(num_samples, -1, 1)], axis=2)

    # load adjacency matrix of GGI
    # The adj_selected_1 matrix contains no diagonal elements and is symmetrical
    adj_1 = sp.load_npz(adj_matrix_file_1)
    adj_mat_1 = adj_1.todense()
    adj_mat_selected_1 = adj_mat_1[high_variance_gene_index, :]
    adj_mat_selected_1 = adj_mat_selected_1[:, high_variance_gene_index]
    adj_selected_1 = sp.csr_matrix(adj_mat_selected_1)  # convert the dense matrix back to sparse matrix

    # load adjacency matrix of Co-expression
    # The adj_selected_2 matrix contains no diagonal elements and is symmetrical
    adj_2 = sp.load_npz(adj_matrix_file_2)
    adj_mat_2 = adj_2.todense()
    adj_mat_selected_2 = adj_mat_2[high_variance_gene_index, :]
    adj_mat_selected_2 = adj_mat_selected_2[:, high_variance_gene_index]
    adj_selected_2 = sp.csr_matrix(adj_mat_selected_2)  # convert the dense matrix back to sparse matrix

    # load adjacency matrix of PPI
    # The adj_selected_3 matrix contains no diagonal elements and is symmetrical
    adj_3 = sp.load_npz(adj_matrix_file_3)
    adj_mat_3 = adj_3.todense()
    adj_mat_selected_3 = adj_mat_3[high_variance_gene_index, :]
    adj_mat_selected_3 = adj_mat_selected_3[:, high_variance_gene_index]
    adj_selected_3 = sp.csr_matrix(adj_mat_selected_3)  # convert the dense matrix back to sparse matrix

    # del features['iCluster']
    shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    # print(shuffle_index.shape)

    return adj_selected_1, adj_selected_2, adj_selected_3, np.asarray(data), labels.to_numpy(), shuffle_index.to_numpy()


def high_variance_expression_gene(expression_variance_path, non_null_path, num_gene, singleton=False):
    gene_variance = pd.read_csv(expression_variance_path, sep='\t', index_col=0, header=0)

    if singleton:
        non_null_row = pd.read_csv(non_null_path, sep=',', header=0)  # [16350, 2]
        gene_variance['id'] = range(gene_variance.shape[0])  # [17945, 2]
        gene_variance_non_null = gene_variance.loc[gene_variance.index.isin(non_null_row['gene']),:]  # [16351, 2]
        gene_list = gene_variance_non_null.nlargest(num_gene, 'variance').index
        gene_variance_non_null.index = gene_variance_non_null['id']
        gene_list_index = gene_variance_non_null.nlargest(num_gene, 'variance').index
    else:
        ## load expression data
        # print(gene_variance['variance'])
        gene_list = gene_variance.nlargest(num_gene, 'variance').index
        gene_variance.index = range(gene_variance.shape[0])
        gene_list_index = gene_variance.nlargest(num_gene, 'variance').index
    return gene_list, gene_list_index


def construct_adjacency_hat(adj):
    """
        :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
        :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # <class 'scipy.sparse.coo.coo_matrix'>
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized
