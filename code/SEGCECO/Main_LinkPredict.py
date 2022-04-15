# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:32:24 2021

@author: sheen
"""

import argparse
from LinkPredict_functions import *


def parse_args(data="MouseD2", epoch=100, lr=0.00001, is_directed=0):
    parser = argparse.ArgumentParser(description="Link prediction with SEAL.")
    parser.add_argument("--data", type=str, help="data name.", default=data)
    parser.add_argument("--epoch", type=int, default=epoch, help="epochs of gnn")
    parser.add_argument("--learning_rate", type=float, default=lr, help="learning rate")
    parser.add_argument("--is_directed",  type=int, default=is_directed, help="use 0, 1 stands for undirected or directed graph")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="ratio of testing set")
    parser.add_argument("--hop", default="auto", help="option: 0, 1, ... or 'auto'.")
    parser.add_argument("--dimension", default=16, type=int, help="number of embedding.")
    return parser.parse_args()


def seal_for_link_predict():
    args = parse_args()
    #classifier(args.data, args.is_directed, args.test_ratio, args.dimension, args.hop, args.learning_rate, epoch=args.epoch)
    #Load graph data
    positive, negative, nodes_size = load_graph_data(args.data, args.is_directed)
    #positive samples: 2148, negative samples: 2148, nodes_size (no of nodes): 297
    
    #Learning node2vec embeddings
    embedding_feature = generate_embeddings(positive, negative, nodes_size, args.test_ratio, args.dimension, args.is_directed)
    #embedding feature shape:  (297, 128)
    
    #Learning attribute features of nodes
    node_features = learning_explicitfeatures(embedding_feature)
       
    #Links to subgraphs
    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size = \
        link2subgraph(positive, negative, nodes_size, args.test_ratio, args.hop, args.is_directed)

    #prepare input for gnn
    D_inverse, A_tilde, Y, X, nodes_size_list, initial_feature_dimension = create_input_for_gnn_fly(
        graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, node_features, None, tags_size)
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
    nodes_size_list_train, nodes_size_list_test = split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    print("data set: ", args.data) #celegan
    print("show configure for gnn.")
    print("number of enclosing sub-graph is: ", len(graphs_adj)) #4296
    #print("size of vertices tags is: ", tags_size) #6
    print("in all enclosing sub-graph, max nodes is %d, min nodes is %d, average node is %.2d." % (
        np.max(nodes_size_list), np.min(nodes_size_list), np.average(nodes_size_list)))
    #in all enclosing sub-graph, max nodes is 100, min nodes is 3, average node is 33.
    top_k=60
    
    test_acc, prediction, pos_scores, aucs = train(X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train,
                     X_test, D_inverse_test, A_tilde_test, Y_test, nodes_size_list_test,
                     top_k, initial_feature_dimension, args.learning_rate, args.epoch)
    auc = metrics.roc_auc_score(y_true=np.squeeze(Y_test), y_score=np.squeeze(pos_scores))
    accuracy = metrics.accuracy_score(np.squeeze(Y_test), np.squeeze(prediction))
    precision = metrics.precision_score(np.squeeze(Y_test), np.squeeze(prediction))
    recall = metrics.recall_score(np.squeeze(Y_test), np.squeeze(prediction))
    f1score = metrics.f1_score(np.squeeze(Y_test), np.squeeze(prediction))
    
    print(args.data)
    print("auc: %f" % auc)
    print("Accuracy: %f" % accuracy)
    print("Precision: %f" % precision)
    print("Recall: %f" % recall)
    print("F1-score: %f" % f1score)

    mean_fpr = np.linspace(0, 1, 100)
    
    #Plot ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true=np.squeeze(Y_test), y_score=np.squeeze(pos_scores))
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    
    #mean_auc = auc(mean_fpr, interp_tpr)
    
    #print("Mean auc is", mean_auc)
    
    np.savetxt("./results/MouseD2/fpr_segceco_MouseD2.txt", np.round(fpr,5), fmt='%.5f')
    
    np.savetxt("./results/MouseD2/tpr_segceco_MouseD2.txt", np.round(tpr,5), fmt='%.5f')
    
    np.savetxt("./results/MouseD2/mean_fpr_segceco_MouseD2.txt", np.round(mean_fpr,5), fmt='%.5f')
    
    np.savetxt("./results/MouseD2/mean_tpr_segceco_MouseD2.txt", np.round(interp_tpr,5), fmt='%.5f')
 
    plt.figure(1)
    plt.plot(mean_fpr, interp_tpr, linestyle='--', label= args.data)    
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title("ROC curve for " + str(args.data))
    plt.savefig("./results/MouseD2/ROC_curve_MouseD2.png")
    # show the plot
    #plt.show()

    mean_recall = np.linspace(0, 1, 100)
     
    #Plot Precision-Recall curve
    prec, rec, threshold = metrics.precision_recall_curve(np.squeeze(Y_test),np.squeeze(pos_scores))
    prs = np.interp(mean_recall, prec, rec)
    
    #mean_auc_pr = auc(mean_recall, prs)
    
    #print("Mean auc PR is", mean_auc_pr)
    
    np.savetxt("./results/MouseD2/prec_segceco_MouseD2.txt", np.round(prec,5), fmt='%.5f')
    
    np.savetxt("./results/MouseD2/rec_segceco_MouseD2.txt", np.round(rec,5), fmt='%.5f')
    
    np.savetxt("./results/MouseD2/mean_recall_segceco_MouseD2.txt", np.round(mean_recall,5), fmt='%.5f')
    
    np.savetxt("./results/MouseD2/mean_precision_segceco_MouseD2.txt", np.round(prs,5), fmt='%.5f') 
    
    plt.figure(2)
    plt.plot(mean_recall, prs, linestyle='--', label= args.data)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.title("Precision Recall Curve for " + str(args.data))
    plt.savefig("./results/MouseD2/Precision_Recall_Curve_MouseD2.png")    
    #plt.show()  
 
    
    return auc, accuracy, precision, recall, f1score    
    
if __name__ == "__main__":
    seal_for_link_predict()