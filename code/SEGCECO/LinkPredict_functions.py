# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:49:34 2021

@author: sheen
"""

from sys import path
import numpy as np
import networkx as nx
from sklearn import metrics
from gensim.models import Word2Vec
from operator import itemgetter
from tqdm import tqdm
import pandas as pd

import time
import tensorflow as tf
import os.path
import glob
import scipy.io as scio
import pickle
import random
import matplotlib.pyplot as plt

GRAPH_CONV_LAYER_CHANNEL = 32
CONV1D_1_OUTPUT = 16
CONV1D_2_OUTPUT = 32
CONV1D_1_FILTER_WIDTH = GRAPH_CONV_LAYER_CHANNEL * 3
CONV1D_2_FILTER_WIDTH = 5
DENSE_NODES = 128
DROP_OUTPUT_RATE = 0.5
LEARNING_RATE_BASE = 0.00004
LEARNING_RATE_DECAY = 0.99
MODEL_SAVE_PATH = "./model"
MODEL_SAVE_NAME = "gnn_model"

#************************ Node2vec ************************************#
class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

#******************************** GNN *************************************
def split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list, rate=0.1):
    print("split training and test data...")
    state = np.random.get_state()
    np.random.shuffle(D_inverse)
    np.random.set_state(state)
    np.random.shuffle(A_tilde)
    np.random.set_state(state)
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    np.random.set_state(state)
    np.random.shuffle(nodes_size_list)
    data_size = Y.shape[0]
    training_set_size, test_set_size = int(data_size * (1 - rate)), int(data_size * rate)
    D_inverse_train, D_inverse_test = D_inverse[: training_set_size], D_inverse[training_set_size:]
    A_tilde_train, A_tilde_test = A_tilde[: training_set_size], A_tilde[training_set_size:]
    X_train, X_test = X[: training_set_size], X[training_set_size:]
    Y_train, Y_test = Y[: training_set_size], Y[training_set_size:]
    nodes_size_list_train, nodes_size_list_test = nodes_size_list[: training_set_size], nodes_size_list[training_set_size:]
    print("about train: positive examples(%d): %s, negative examples: %s."
          % (training_set_size, np.sum(Y_train == 1), np.sum(Y_train == 0)))
    print("about test: positive examples(%d): %s, negative examples: %s."
          % (test_set_size, np.sum(Y_test == 1), np.sum(Y_test == 0)))
    return D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
           nodes_size_list_train, nodes_size_list_test

def train(X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train,
        X_test, D_inverse_test, A_tilde_test, Y_test, nodes_size_list_test,
        top_k, initial_channels, learning_rate=0.00001, epoch=100, data_name="mutag", debug=False):
        

    # placeholder
    D_inverse_pl = tf.placeholder(dtype=tf.float32, shape=[None, None])
    A_tilde_pl = tf.placeholder(dtype=tf.float32, shape=[None, None])
    X_pl = tf.placeholder(dtype=tf.float32, shape=[None, initial_channels])
    Y_pl = tf.placeholder(dtype=tf.int32, shape=[1], name="Y-placeholder")
    node_size_pl = tf.placeholder(dtype=tf.int32, shape=[], name="node-size-placeholder")
    is_train = tf.placeholder(dtype=tf.uint8, shape=[], name="is-train-or-test")
    
    print("Training data shape", X_train.shape)
    print("Training data shape", Y_train.shape)
    print("Testing data shape", X_test.shape)
    print("Testing data shape", X_test.shape)

    # trainable parameters of graph convolution layer
    graph_weight_1 = tf.Variable(tf.truncated_normal(shape=[initial_channels, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32))
    graph_weight_2 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32))
    graph_weight_3 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32))
    graph_weight_4 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, 1], stddev=0.1, dtype=tf.float32))

    # GRAPH CONVOLUTION LAYER
    gl_1_XxW = tf.matmul(X_pl, graph_weight_1)
    gl_1_AxXxW = tf.matmul(A_tilde_pl, gl_1_XxW)
    Z_1 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_1_AxXxW))
    gl_2_XxW = tf.matmul(Z_1, graph_weight_2)
    gl_2_AxXxW = tf.matmul(A_tilde_pl, gl_2_XxW)
    Z_2 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_2_AxXxW))
    gl_3_XxW = tf.matmul(Z_2, graph_weight_3)
    gl_3_AxXxW = tf.matmul(A_tilde_pl, gl_3_XxW)
    Z_3 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_3_AxXxW))
    gl_4_XxW = tf.matmul(Z_3, graph_weight_4)
    gl_4_AxXxW = tf.matmul(A_tilde_pl, gl_4_XxW)
    Z_4 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_4_AxXxW))
    graph_conv_output = tf.concat([Z_1, Z_2, Z_3], axis=1)  # shape=(node_size/None, 32+32+32)

    def variable_summary(var):
        var_mean = tf.reduce_mean(var)
        var_variance = tf.square(tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))
        var_max = tf.reduce_max(var)
        var_min = tf.reduce_min(var)
        return var_mean, var_variance, var_max, var_min

    if debug:
        var_mean, var_variance, var_max, var_min = variable_summary(graph_weight_1)

    # SORT POOLING LAYER
    nodes_size_list = list(nodes_size_list_train) + list(nodes_size_list_test)
    threshold_k = int(np.percentile(nodes_size_list, top_k))
    print("%s%% graphs have nodes less then %s." % (top_k, threshold_k))

    graph_conv_output_stored = tf.gather(graph_conv_output, tf.nn.top_k(Z_4[:, 0], node_size_pl).indices)
    graph_conv_output_top_k = tf.cond(tf.less(node_size_pl, threshold_k),
                                      lambda: tf.concat(axis=0,
                                                        values=[graph_conv_output_stored,
                                                                tf.zeros(dtype=tf.float32,
                                                                         shape=[threshold_k-node_size_pl,
                                                                                GRAPH_CONV_LAYER_CHANNEL*3])]),
                                      lambda: tf.slice(graph_conv_output_stored, begin=[0, 0], size=[threshold_k, -1]))

    # FLATTEN LAYER
    graph_conv_output_flatten = tf.reshape(graph_conv_output_top_k, shape=[1, GRAPH_CONV_LAYER_CHANNEL*3*threshold_k, 1])
    assert graph_conv_output_flatten.shape == [1, GRAPH_CONV_LAYER_CHANNEL*3*threshold_k, 1]

    # 1-D CONVOLUTION LAYER 1:
    # kernel = (filter_width, in_channel, out_channel)
    conv1d_kernel_1 = tf.Variable(tf.truncated_normal(shape=[CONV1D_1_FILTER_WIDTH, 1, CONV1D_1_OUTPUT], stddev=0.1, dtype=tf.float32))
    conv_1d_a = tf.nn.conv1d(graph_conv_output_flatten, conv1d_kernel_1, stride=CONV1D_1_FILTER_WIDTH, padding="VALID")
    assert conv_1d_a.shape == [1, threshold_k, CONV1D_1_OUTPUT]
    # 1-D CONVOLUTION LAYER 2:
    conv1d_kernel_2 = tf.Variable(tf.truncated_normal(shape=[CONV1D_2_FILTER_WIDTH, CONV1D_1_OUTPUT, CONV1D_2_OUTPUT], stddev=0.1, dtype=tf.float32))
    conv_1d_b = tf.nn.conv1d(conv_1d_a, conv1d_kernel_2, stride=1, padding="VALID")
    assert conv_1d_b.shape == [1, threshold_k - CONV1D_2_FILTER_WIDTH + 1, CONV1D_2_OUTPUT]
    conv_output_flatten = tf.layers.flatten(conv_1d_b)

    # DENSE LAYER
    weight_1 = tf.Variable(tf.truncated_normal(shape=[int(conv_output_flatten.shape[1]), DENSE_NODES], stddev=0.1))
    bias_1 = tf.Variable(tf.zeros(shape=[DENSE_NODES]))
    dense_z = tf.nn.relu(tf.matmul(conv_output_flatten, weight_1) + bias_1)
    if is_train == 1:
        dense_z = tf.layers.dropout(dense_z, DROP_OUTPUT_RATE)

    weight_2 = tf.Variable(tf.truncated_normal(shape=[DENSE_NODES, 2]))
    bias_2 = tf.Variable(tf.zeros(shape=[2]))
    pre_y = tf.matmul(dense_z, weight_2) + bias_2
    pos_score = tf.nn.softmax(pre_y)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_pl, logits=pre_y))
    global_step = tf.Variable(0, trainable=False)

    train_data_size, test_data_size = X_train.shape[0], X_test.shape[0]

    # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, train_data_size, LEARNING_RATE_DECAY, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
    # saver = tf.train.Saver()
    
    aucs = []
    accs = []

    with tf.Session() as sess:
        print("start training gnn.")
        print("learning rate: %f. epoch: %d." % (learning_rate, epoch))
        start_t = time.time()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch):
            batch_index = 0
            for _ in tqdm(range(train_data_size)):
                batch_index = batch_index + 1 if batch_index < train_data_size - 1 else 0
                feed_dict = {D_inverse_pl: D_inverse_train[batch_index],
                             A_tilde_pl: A_tilde_train[batch_index],
                             X_pl: X_train[batch_index],
                             Y_pl: Y_train[batch_index],
                             node_size_pl: nodes_size_list_train[batch_index],
                             is_train: 1
                             }
                loss_value, _, _ = sess.run([loss, train_op, global_step], feed_dict=feed_dict)

            train_acc = 0
            for i in range(train_data_size):
                feed_dict = {D_inverse_pl: D_inverse_train[i], A_tilde_pl: A_tilde_train[i],
                             X_pl: X_train[i], Y_pl: Y_train[i], node_size_pl: nodes_size_list_train[i], is_train: 0}
                pre_y_value = sess.run(pre_y, feed_dict=feed_dict)
                if np.argmax(pre_y_value, 1) == Y_train[i]:
                    train_acc += 1
            train_acc = train_acc / train_data_size

            test_acc, prediction, scores = 0, [], []
            for i in range(test_data_size):
                feed_dict = {D_inverse_pl: D_inverse_test[i], A_tilde_pl: A_tilde_test[i],
                             X_pl: X_test[i], Y_pl: Y_test[i], node_size_pl: nodes_size_list_test[i], is_train: 0}
                pre_y_value, pos_score_value = sess.run([pre_y, pos_score], feed_dict=feed_dict)
                prediction.append(np.argmax(pre_y_value, 1))
                scores.append(pos_score_value[0][1])
                if np.argmax(pre_y_value, 1) == Y_test[i]:
                    test_acc += 1
            test_acc = test_acc / test_data_size
            if debug:
                mean_value, var_value, max_value, min_value = sess.run([var_mean, var_variance, var_max, var_min], feed_dict=feed_dict)
                print("\t\tdebug: mean: %f, variance: %f, max: %f, min: %f." %
                      (mean_value, var_value, max_value, min_value))
            
            print("After %5s epoch, the loss is %f, training acc %f, test acc %f, auc is %f."
                  % (epoch, loss_value, train_acc, test_acc, metrics.roc_auc_score(y_true=np.squeeze(Y_test), y_score=np.squeeze(scores))))
            accs.append(test_acc)
            aucs.append(metrics.roc_auc_score(y_true=np.squeeze(Y_test), y_score=np.squeeze(scores)))
            # saver.save(sess, os.path.join(MODEL_SAVE_PATH, data_name, MODEL_SAVE_NAME), global_step)
        end_t = time.time()
        print("time consumption: ", end_t - start_t)
    return accs, prediction, scores, aucs

def load_graph_data(dataset_name, network_type):
    """
    :param dataset_name: 
    :param network_type: use 0 and 1 stands for undirected or directed graph, respectively
    :return: 
    """
    print("Load Data...")
    file_path = "D:/Masters/My_thesis/Code/raw_data/" + dataset_name + ".txt"
    positive = np.loadtxt(file_path, dtype=int, usecols=(0, 1))
  
    G = nx.Graph() if network_type == 0 else nx.DiGraph()
    G.add_edges_from(positive)
    print(nx.info(G))
    
    #add negative edges
    negative_all = list(nx.non_edges(G))
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:len(positive)])
    print("Positive Edges: %d, Negative Edges: %d." % (len(positive), len(negative)))
    #positve examples: 2148, negative examples: 2148.
    np.random.shuffle(positive)
    if np.min(positive) == 1:
        positive -= 1
        negative -= 1
    return positive, negative, len(G.nodes())

def generate_embeddings(positive, negative, network_size,test_ratio, dimension, network_type, negative_injection=True):
    """
    :param positive: ndarray, from 'load_data', all positive edges
    :param negative: ndarray, from 'load_data', all negative edges
    :param network_size: scalar, total number of nodes in the network
    :param test_ratio: proportion of the test set 
    :param dimension: size of the node2vec
    :param network_type: directed or undirected
    :param negative_injection: add negative edges to learn word embedding
    :return: 
    """
    print("learning embedding...")
    # used training data only
    test_size = int(test_ratio * positive.shape[0])
    train_posi, train_nega = positive[:-test_size], negative[:-test_size]
    # negative injection
    A = nx.Graph() if network_type == 0 else nx.DiGraph()
    A.add_weighted_edges_from(np.concatenate([train_posi, np.ones(shape=[train_posi.shape[0], 1], dtype=np.int8)], axis=1))
    if negative_injection:
        A.add_weighted_edges_from(np.concatenate([train_nega, np.ones(shape=[train_nega.shape[0], 1], dtype=np.int8)], axis=1))
    # node2vec
    G = Graph(A, is_directed=False if network_type == 0 else True, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimension, window=10, min_count=0, sg=1, workers=8, iter=1)
    wv = model.wv
    embedding_feature, empty_indices, avg_feature = np.zeros([network_size, dimension]), [], 0
    for i in range(network_size):
        if str(i) in wv:
            embedding_feature[i] = wv.word_vec(str(i))
            avg_feature += wv.word_vec(str(i))
        else:
            empty_indices.append(i)
    embedding_feature[empty_indices] = avg_feature / (network_size - len(empty_indices))
    print("Node2vec embedding feature shape: ", embedding_feature.shape)
    return embedding_feature

def learning_explicitfeatures(embedding_feature):
    """
    :param embedding_feature: ndarray, from 'generate_embeddings', node2vec embeddings of the graph with positive and negative edges
    :return: 
    """
    attributes_mat1 = pd.read_csv("./data/MouseD2/attributes_IG_MouseD2.csv")
    attributes_mat = attributes_mat1.iloc[:297 , :]
    if attributes_mat is not None:
        if embedding_feature is not None:
            #node_features = np.concatenate([embedding_feature, attributes_mat.iloc[:297]], axis=1)
            node_features = np.concatenate([embedding_feature, attributes_mat], axis=1)
        else:
            node_features = attributes_mat
    print("Node Features shape: ", node_features.shape) 
    return node_features

def link2subgraph(positive, negative, nodes_size, test_ratio, hop, network_type, max_neighbors=100):
    """
    :param positive: ndarray, from 'load_data', all positive edges
    :param negative: ndarray, from 'load_data', all negative edges
    :param nodes_size: int, scalar, nodes size in the network
    :param test_ratio: float, scalar, proportion of the test set 
    :param hop: option: 0, 1, 2, ..., or 'auto'
    :param network_type: directed or undirected
    :param max_neighbors: 
    :return: 
    """
    print("extract enclosing subgraph...")
    test_size = int(len(positive) * test_ratio)
    train_pos, test_pos = positive[:-test_size], positive[-test_size:]
    train_neg, test_neg = negative[:-test_size], negative[-test_size:]

    A = np.zeros([nodes_size, nodes_size], dtype=np.uint8)
    A[train_pos[:, 0], train_pos[:, 1]] = 1
    if network_type == 0:
        A[train_pos[:, 1], train_pos[:, 0]] = 1

    def calculate_auc(scores, test_pos, test_neg):
        pos_scores = scores[test_pos[:, 0], test_pos[:, 1]]
        neg_scores = scores[test_neg[:, 0], test_neg[:, 1]]
        s = np.concatenate([pos_scores, neg_scores])
        y = np.concatenate([np.ones(len(test_pos), dtype=np.int8), np.zeros(len(test_neg), dtype=np.int8)])
        assert len(s) == len(y)
        auc = metrics.roc_auc_score(y_true=y, y_score=s)
        return auc

    # determine the h value
    if hop == "auto":
        def cn():
            return np.matmul(A, A)
        def aa():
            A_ = A / np.log(A.sum(axis=1))
            A_[np.isnan(A_)] = 0
            A_[np.isinf(A_)] = 0
            return A.dot(A_)
        cn_scores, aa_scores = cn(), aa()
        cn_auc = calculate_auc(cn_scores, test_pos, test_neg)
        aa_auc = calculate_auc(aa_scores, test_pos, test_neg)
        if cn_auc > aa_auc:
            print("cn(first order heuristic): %f > aa(second order heuristic) %f." % (cn_auc, aa_auc))
            hop = 1
        else:
            print("aa(second order heuristic): %f > cn(first order heuristic) %f. " % (aa_auc, cn_auc))
            hop = 2

    print("hop = %s." % hop)

    # extract the subgraph for (positive, negative)
    G = nx.Graph() if network_type == 0 else nx.DiGraph()
    G.add_nodes_from(set(positive[:, 0]) | set(positive[:, 1]) | set(negative[:, 0]) | set(negative[:, 1]))
    # G.add_nodes_from(set(sum(positive.tolist(), [])) | set(sum(negative.tolist(), [])))
    G.add_edges_from(train_pos)

    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes = [], [], [], [], []
    for graph_label, data in enumerate([negative, positive]):
        print("for %s. " % "negative" if graph_label == 0 else "positive")
        for node_pair in tqdm(data):
            sub_nodes, sub_adj, vertex_tag = extract_subgraph(node_pair, G, A, hop, network_type, max_neighbors)
            graphs_adj.append(sub_adj)
            vertex_tags.append(vertex_tag)
            node_size_list.append(len(vertex_tag))
            sub_graphs_nodes.append(sub_nodes)
    assert len(graphs_adj) == len(vertex_tags) == len(node_size_list)
    labels = np.concatenate([np.zeros(len(negative), dtype=np.uint8), np.ones(len(positive), dtype=np.uint8)]).reshape(-1, 1)

    # vertex_tags_set = list(set(sum(vertex_tags, [])))
    vertex_tags_set = set()
    for tags in vertex_tags:
        vertex_tags_set = vertex_tags_set.union(set(tags))
    vertex_tags_set = list(vertex_tags_set)
    tags_size = len(vertex_tags_set)
    print("tight the vertices tags.")
    if set(range(len(vertex_tags_set))) != set(vertex_tags_set):
        vertex_map = dict([(x, vertex_tags_set.index(x)) for x in vertex_tags_set])
        for index, graph_tag in tqdm(enumerate(vertex_tags)):
            vertex_tags[index] = list(itemgetter(*graph_tag)(vertex_map))
    return graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size


def extract_subgraph(node_pair, G, A, hop, network_type, max_neighbors):
    """
    :param node_pair:  (vertex_start, vertex_end)
    :param G:  nx object from the positive edges
    :param A:  equivalent to the G, adj matrix of G
    :param hop:
    :param network_type:
    :param max_neighbors: 
    :return: 
        sub_graph_nodes: use for select the embedding feature
        sub_graph_adj: adjacent matrix of the enclosing sub-graph
        vertex_tag: node type information from the labeling algorithm
    """
    sub_graph_nodes = set(node_pair)
    nodes = list(node_pair)

    for i in range(int(hop)):
        #np.random.shuffle(nodes)
        for node in nodes:
            neighbors = list(nx.neighbors(G, node))
            if len(sub_graph_nodes) + len(neighbors) < max_neighbors:
                sub_graph_nodes = sub_graph_nodes.union(neighbors)
            else:
                np.random.shuffle(neighbors)
                sub_graph_nodes = sub_graph_nodes.union(neighbors[:max_neighbors - len(sub_graph_nodes)])
                break
        nodes = sub_graph_nodes - set(nodes)
    sub_graph_nodes.remove(node_pair[0])
    if node_pair[0] != node_pair[1]:
        sub_graph_nodes.remove(node_pair[1])
    sub_graph_nodes = [node_pair[0], node_pair[1]] + list(sub_graph_nodes)
    sub_graph_adj = A[sub_graph_nodes, :][:, sub_graph_nodes]
    sub_graph_adj[0][1] = sub_graph_adj[1][0] = 0

    # labeling(coloring/tagging)
    vertex_tag = node_labeling(sub_graph_adj, network_type)
    return sub_graph_nodes, sub_graph_adj, vertex_tag


def node_labeling(graph_adj, network_type):
    nodes_size = len(graph_adj)
    G = nx.Graph(data=graph_adj) if network_type == 0 else nx.DiGraph(data=graph_adj)
    if len(G.nodes()) == 0:
        return [1, 1]
    tags = []
    for node in range(2, nodes_size):
        try:
            dx = nx.shortest_path_length(G, 0, node)
            dy = nx.shortest_path_length(G, 1, node)
        except nx.NetworkXNoPath:
            tags.append(0)
            continue
        d = dx + dy
        div, mod = np.divmod(d, 2)
        tag = 1 + np.min([dx, dy]) + div * (div + mod - 1)
        tags.append(tag)
    return [1, 1] + tags


def create_input_for_gnn_fly(graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes,
                             embedding_feature, explicit_feature, tags_size):
    print("create input for gnn on fly, (skipping I/O operation)")
    # graphs, nodes_size_list, labels = data["graphs"], data["nodes_size_list"], data["labels"]

    # 1 - prepare Y
    Y = np.where(np.reshape(labels, [-1, 1]) == 1, 1, 0)
    print("positive examples: %d, negative examples: %d." % (np.sum(Y == 0), np.sum(Y == 1)))
    # 2 - prepare A_title
    # graphs_adj is A_title in the formular of Graph Convolution layer
    # add eye to A_title
    for index, x in enumerate(graphs_adj):
        graphs_adj[index] = x + np.eye(x.shape[0], dtype=np.uint8)
    # 3 - prepare D_inverse
    D_inverse = []
    for x in graphs_adj:
        D_inverse.append(np.linalg.inv(np.diag(np.sum(x, axis=1))))
    # 4 - prepare X
    X, initial_feature_channels = [], 0

    def convert_to_one_hot(y, C):
        return np.eye(C, dtype=np.uint8)[y.reshape(-1)]

    if vertex_tags is not None:
        initial_feature_channels = tags_size
        print("X: one-hot vertex tag, tag size %d." % initial_feature_channels)
        for tag in vertex_tags:
            x = convert_to_one_hot(np.array(tag), initial_feature_channels)
            X.append(x)
    else:
        print("X: normalized node degree.")
        for graph in graphs_adj:
            degree_total = np.sum(graph, axis=1)
            X.append(np.divide(degree_total, np.sum(degree_total)).reshape(-1, 1))
        initial_feature_channels = 1
    X = np.array(X)
    if embedding_feature is not None:
        print("embedding feature has considered")
        # build embedding for enclosing sub-graph
        sub_graph_emb = []
        for sub_nodes in sub_graphs_nodes:
            sub_graph_emb.append(embedding_feature[sub_nodes])
        for i in range(len(X)):
            X[i] = np.concatenate([X[i], sub_graph_emb[i]], axis=1)
        initial_feature_channels = len(X[0][0])
    if explicit_feature is not None:
        initial_feature_channels = len(X[0][0])
        pass
    print("so, initial feature channels: ", initial_feature_channels)
    return np.array(D_inverse), graphs_adj, Y, X, node_size_list, initial_feature_channels  # ps, graph_adj is A_title
    
    
