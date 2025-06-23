"""
paper: DBGCN: Dual-branch Graph Convolutional Network for organ instance inference on sparsely labeled 3D plant data
file:  utils.py
about: Tools for model utilization
author: Zhaoyi Zhou
date: 2025-6-19
"""

import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = [str(i) for i in range(labels.shape[0])]
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(contentpath=".csv",adjpath=".txt"):
    print('Loading {} dataset...'.format(contentpath))

    idx_features_labels = np.genfromtxt(contentpath,dtype=np.dtype(str))
    print("read csv alreadyly")
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    labels = encode_onehot(idx_features_labels[:, -1])  # one-hot label
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(adjpath, delimiter='=',
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = (adj + sp.eye(adj.shape[0]))
    # transfer to torch
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct0 = preds.eq(labels).double()
    correct = correct0.sum()
    return correct / len(labels), preds

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)