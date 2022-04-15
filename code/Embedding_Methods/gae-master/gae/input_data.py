import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_dataset():    
    #G = nx.read_edgelist('D:/Masters/My_thesis/Code/edgelist_encoded_D1.edgelist', nodetype=int, create_using=nx.Graph())
    G = nx.read_edgelist('./data/MouseD2/edgelist_encoded_MouseD2.edgelist', nodetype=int, create_using=nx.Graph())

    G = G.to_undirected()
    adj = nx.adjacency_matrix(G)
    features = np.identity(adj.shape[0])
    features = nx.adjacency_matrix(nx.from_numpy_matrix(features))

    return adj, features
