# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:01:23 2022

@author: sheen
"""

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from gensim.models import KeyedVectors
from sklearn.model_selection import KFold 

import os
os.getcwd()
#os.chdir("D:/Masters/My_thesis/Code")

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}

def edges_to_features(edge_list, edge_function, dimensions):
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
        wvecs = load_embeddings()
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, dimensions), dtype='f')

        # Iterate over edges
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = np.asarray(wvecs[str(v1)])
            emb2 = np.asarray(wvecs[str(v2)])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec

def load_embeddings():    
     #emb = KeyedVectors.load_word2vec_format("./code/Embedding_Results/MouseD2/Node2vec_16_D2.emd", binary=False)
     #emb = KeyedVectors.load_word2vec_format("./code/Embedding_Results/MouseD2/LINE_16_D2.emd", binary=False)
     emb = KeyedVectors.load_word2vec_format("./code/Embedding_Results/MouseD2/DeepWalk_16_MouseD2.emd", binary=False)
     return emb
   
def test_edge_functions():
    #dimensions = 128
    dimensions = 16

    #G = nx.read_edgelist('./edgelist_encoded_D1.edgelist', nodetype=int, create_using=nx.Graph())
    #G = nx.read_edgelist('./raw_data/Yeast.txt', nodetype=int, create_using=nx.Graph())
    G = nx.read_edgelist('./data/MouseD2/edgelist_encoded_MouseD2.edgelist', nodetype=int, create_using=nx.Graph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G = G.to_undirected()
    #'Name: \nType: Graph\nNumber of nodes: 1930\nNumber of edges: 33941\nAverage degree:  35.1720'
    
    positive = list(G.edges())
    #sample negative     
    negative_all = list(nx.non_edges(G))
    np.random.shuffle(negative_all)
    negative = negative_all[:len(positive)]
    np.random.shuffle(positive)
    
    print(nx.info(G))
    print("Positive edges: %d, Negative edges: %d." % (len(positive), len(negative)))
    positive = np.array(positive)
    negative = np.array(negative)
    aucs = {name: [] for name in edge_functions}
    mean_tprs = {name: [] for name in edge_functions}
    mean_fpr = np.linspace(0, 1, 100)
    
    tprs = {name: [] for name in edge_functions}
    fprs = {name: [] for name in edge_functions}
    
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(positive):
        print("Training data shape:", train_index.shape, "Testing data shape:", test_index.shape)
        train_pos, test_pos = positive[train_index], positive[test_index]
        train_neg, test_neg = negative[train_index], negative[test_index]
        
        print(train_pos.shape)
        print(test_pos.shape) #(30546, 2)
        print(train_neg.shape)
        print(test_neg.shape) #(3395, 2)
        
        train_pos = np.array(train_pos)
        train_neg = np.array(train_neg)
        test_pos = np.array(test_pos)
        test_neg = np.array(test_neg)
        
        train_edges = np.concatenate([train_pos, train_neg]) #47518
        test_edges = np.concatenate([test_pos, test_neg]) #20364
    
        train_labels = np.zeros(len(train_edges))
        train_labels[:len(train_pos)] = 1 #47518
    
        test_labels = np.zeros(len(test_edges)) #20364
        test_labels[:len(test_pos)] = 1
        
        print("Iteration")
         
        for edge_fn_name, edge_fn in edge_functions.items():
            # Calculate edge embeddings using binary function
            edge_features_train = edges_to_features(train_edges, edge_fn, dimensions)
            edge_features_test = edges_to_features(test_edges, edge_fn, dimensions)

            # LINEar classifier
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
            
            fpr, tpr, thresholds = metrics.roc_curve(test_labels, preds)   
            fprs[edge_fn_name].append(fpr)
            tprs[edge_fn_name].append(tpr)
            
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            mean_tprs[edge_fn_name].append(interp_tpr)   

            # Test classifier
            #auc_test = metrics.roc_auc_score(clf, edge_features_test, test_labels)
            aucs[edge_fn_name].append(auc_test)
            print("[%s] AUC: %f" % (edge_fn_name, auc_test))
   
    return aucs, mean_tprs, mean_fpr, fprs, tprs
 
if __name__ == "__main__":
    aucs, mean_tprs, mean_fpr, fprs, tprs = test_edge_functions()
    
    print("Edge function test performance (AUC):")
    for edge_name in aucs:
        auc_mean = np.mean(aucs[edge_name])
        auc_std = np.std(aucs[edge_name])
        print("[%s] mean: %.4g +/- %.3g" % (edge_name, auc_mean, auc_std))
        
    tpr_hadamard = np.mean(tprs['hadamard'], axis=0)
    np.savetxt("./Results/MouseD2/DeepWalk/tpr_hadamard.txt", tpr_hadamard, fmt='%.5f')
    
    tpr_l1 = np.mean(tprs['l1'], axis=0)
    np.savetxt("./Results/MouseD2/DeepWalk/tpr_l1.txt", tpr_l1, fmt='%.5f')
    
    tpr_l2 = np.mean(tprs['l2'], axis=0)
    np.savetxt("./Results/MouseD2/DeepWalk/tpr_l2.txt", tpr_l2, fmt='%.5f')
    
    tpr_average = np.mean(tprs['average'], axis=0)
    np.savetxt("./Results/MouseD2/DeepWalk/tpr_average.txt", tpr_average, fmt='%.5f')
    
    fpr_hadamard = np.mean(fprs['hadamard'], axis=0)
    np.savetxt("./Results/MouseD2/DeepWalk/fpr_hadamard.txt", fpr_hadamard, fmt='%.5f')
    
    fpr_l1 = np.mean(fprs['l1'], axis=0)
    np.savetxt("./Results/MouseD2/DeepWalk/fpr_l1.txt", fpr_l1, fmt='%.5f')
    
    fpr_l2 = np.mean(fprs['l2'], axis=0)
    np.savetxt("./Results/MouseD2/DeepWalk/fpr_l2.txt", fpr_l2, fmt='%.5f')
    
    fpr_average = np.mean(fprs['average'], axis=0)
    np.savetxt("./Results/MouseD2/DeepWalk/fpr_average.txt", fpr_average, fmt='%.5f')
        
    np.savetxt("./Results/MouseD2/mean_fpr.txt", mean_fpr, fmt='%.5f')
              
    mean_tpr_hadamard = np.mean(mean_tprs['hadamard'], axis=0)
    mean_tpr_hadamard[-1] = 1.0
    np.savetxt("./Results/MouseD2/DeepWalk/mean_tpr_hadamard.txt", mean_tpr_hadamard, fmt='%.5f')
        
    mean_tpr_l1 = np.mean(mean_tprs['l1'], axis=0)
    mean_tpr_l1[-1] = 1.0
    np.savetxt("./Results/MouseD2/DeepWalk/mean_tpr_l1.txt", mean_tpr_l1, fmt='%.5f')
        
    mean_tpr_l2 = np.mean(mean_tprs['l2'], axis=0)
    mean_tpr_l2[-1] = 1.0
    np.savetxt("./Results/MouseD2/DeepWalk/mean_tpr_l2.txt", mean_tpr_l2, fmt='%.5f')
        
    mean_tpr_average = np.mean(mean_tprs['average'], axis=0)
    mean_tpr_average[-1] = 1.0
    np.savetxt("./Results/MouseD2/DeepWalk/mean_tpr_average.txt", mean_tpr_average, fmt='%.5f')
        

               
   print("Edge function TPR:")
   for edge_name in tprs:
        print(edge_name) 
        mean_tpr = np.mean(tprs[edge_name], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        
        print(mean_tpr, mean_fpr)
        
        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        std_auc = np.std(aucs[edge_name])
        plt.plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
                )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
