from enum import Enum

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import networkx as nx
# from torch_geometric.datasets import Planetoid
from networkx import to_scipy_sparse_matrix
# from torch_geometric.utils import to_scipy_sparse_matrix


def get_adjacency_matrix(edge_index):
    # working with scipy sparse since current PyTorch version doesn't support sparse x sparse multiplication
    edge_index = edge_index.to(int).tolist()
    edge_index = [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_index[0]))]
    G = nx.Graph(edge_index)
    adj = to_scipy_sparse_matrix(G)
    adj += sparse.eye(adj.shape[0])  # add self loops
    degree_for_norm = sparse.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())  # D^(-0.5)
    adj_hat_csr = degree_for_norm.dot(adj.dot(degree_for_norm))  # D^(-0.5) * A * D^(-0.5)
    adj_hat_coo = adj_hat_csr.tocoo().astype(np.float32)
    # to torch sparse matrix
    indices = torch.from_numpy(np.vstack((adj_hat_coo.row, adj_hat_coo.col)).astype(np.int64))
    values = torch.from_numpy(adj_hat_coo.data)
    adjacency_matrix = torch.sparse_coo_tensor(indices, values, torch.Size(adj_hat_coo.shape))
    return adjacency_matrix.to_dense(), adj_hat_csr


def get_laplacian_matrix(adjacency_matrix_csr: sparse.csr_matrix):
    # since adjacency_matrix_csr is already in form D^(-0.5) * A * D^(-0.5), we can simply get normalized laplacian by:
    laplacian = sparse.eye(adjacency_matrix_csr.shape[0]) - adjacency_matrix_csr
    # rescaling laplacian
    max_eigenval = sparse.linalg.eigsh(laplacian, k=1, which='LM', return_eigenvectors=False)[0]
    laplacian = 2 * laplacian / max_eigenval - sparse.eye(adjacency_matrix_csr.shape[0])
    # to torch sparse matrix
    laplacian = laplacian.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((laplacian.row, laplacian.col)).astype(np.int64))
    values = torch.from_numpy(laplacian.data)
    laplacian_matrix = torch.sparse_coo_tensor(indices, values, torch.Size(laplacian.shape))
    return laplacian_matrix
