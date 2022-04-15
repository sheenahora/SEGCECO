# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:00:15 2022

@author: sheen
"""


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.manifold import spectral_embedding
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from EdgeSplit import *

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}

def edges_to_features(edge_list, edge_function, dimensions, emb):
        """
        Given a list of edge lists and a list of labels, create
        an edge feature array using binary_edge_function and
        create a label array matching the label in the list to all
        edges in the corresponding edge list
        :param edge_function:
            Function of two arguments taking the node features and returning
            an edge feature of given dimension
        :param dimension:
            Size of returned edge feature vector, if None defaults to
            node feature size.
        :param k:
            Partition number. If None use all positive & negative edges
        :return:
            feature_vec (n, dimensions), label_vec (n)
        """
        wvecs = emb
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, dimensions), dtype='f')

        # Iterate over edges
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = np.asarray(wvecs[v1])
            emb2 = np.asarray(wvecs[v2])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec
    
def main():
    G = nx.read_edgelist('D:/Masters/My_thesis/Code/edgelist_encoded_D1.edgelist', nodetype=int, create_using=nx.Graph())
    #G = nx.read_edgelist('D:/Masters/My_thesis/Code/raw_data/Yeast.txt', nodetype=int, create_using=nx.Graph())
       
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G = G.to_undirected()
    print(nx.info(G))
    #'Name: \nType: Graph\nNumber of nodes: 1930\nNumber of edges: 33941\nAverage degree:  35.1720'

    np.random.seed(0) # make sure train-test split is consistent between notebooks
    adj_sparse = nx.to_scipy_sparse_matrix(G)
    
    # Perform train-test split
    adj_train, train_edges_pos, train_edges_false, val_edges_pos, val_edges_false, \
        test_edges_pos, test_edges_false = train_test_val_split(adj_sparse, test_frac=.3, val_frac=.1)
    
    g_train = nx.from_scipy_sparse_matrix(adj_train)
    
    # Inspect train/test split
    print("Total nodes:", adj_sparse.shape[0])
    print("Total edges:", int(adj_sparse.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
    print("Training edges (positive):", len(train_edges_pos))
    print("Training edges (negative):", len(train_edges_false))
    print("Validation edges (positive):", len(val_edges_pos))
    print("Validation edges (negative):", len(val_edges_false))
    print("Test edges (positive):", len(test_edges_pos))
    print("Test edges (negative):", len(test_edges_false))
    
    train_pos = np.array(train_edges_pos)
    train_neg = np.array(train_edges_false)
    test_pos = np.array(test_edges_pos)
    test_neg = np.array(test_edges_false)
        
    train_edges = np.concatenate([train_pos, train_neg]) #47518
    test_edges = np.concatenate([test_pos, test_neg]) #20364
    
    train_labels = np.zeros(len(train_edges))
    train_labels[:len(train_pos)] = 1 #47518
    
    test_labels = np.zeros(len(test_edges)) #20364
    test_labels[:len(test_pos)] = 1
    
    # Get spectral embeddings (16-dim)
    emb = spectral_embedding(adj_sparse, n_components=16, random_state=0)
   
    aucs = {name: [] for name in edge_functions}
    dimensions = 16
    num_iterations = 10
    for i in range(0,num_iterations):
        print("Iteration: ", i)
        for edge_fn_name, edge_fn in edge_functions.items():
            # Calculate edge embeddings using binary function
            edge_features_train = edges_to_features(train_edges, edge_fn, dimensions, emb)
            edge_features_test = edges_to_features(test_edges, edge_fn, dimensions, emb)

            # Linear classifier
            scaler = StandardScaler()
            lin_clf = LogisticRegression(C=1)
            clf = pipeline.make_pipeline(scaler, lin_clf)
            # Train classifier
            clf.fit(edge_features_train, train_labels)
            #auc_train = metrics.roc_auc_score(clf, edge_features_train, train_labels)
            preds = clf.predict(edge_features_test)
            #probs = clf.predict_proba(edge_features_test)
            #preds = probs[:,1]
            auc_test = metrics.roc_auc_score(test_labels, preds)

            # Test classifier
            #auc_test = metrics.roc_auc_score(clf, edge_features_test, test_labels)
            aucs[edge_fn_name].append(auc_test)
            print("[%s] AUC: %f" % (edge_fn_name, auc_test))
        print("Edge function test performance (AUC):")
    
    for edge_name in aucs:
        auc_mean = np.mean(aucs[edge_name])
        auc_std = np.std(aucs[edge_name])
        print("[%s] mean: %.4g +/- %.3g" % (edge_name, auc_mean, auc_std))
    
    return aucs
 
if __name__ == "__main__":
    main()

    