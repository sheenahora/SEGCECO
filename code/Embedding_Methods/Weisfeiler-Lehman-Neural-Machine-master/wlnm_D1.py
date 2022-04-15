# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:19:31 2022

@author: sheen
"""

import networkx as nx
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import math
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from functools import partial


print("load data...")
#G = nx.read_edgelist('D:/Masters/My_thesis/Code/edgelist_encoded_D1.edgelist', nodetype=int, create_using=nx.Graph())
G = nx.read_edgelist('D:/Masters/My_thesis/Code/raw_data/PB.txt', nodetype=int, create_using=nx.Graph())
      
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1
G = G.to_undirected()
print(nx.info(G))
#Graph with 1930 nodes and 33941 edges

G.number_of_nodes(), G.number_of_edges()
#(1930, 33941)

# ## Dividing the network into the training and test networks

G_train = G.copy()
#'Graph with 1930 nodes and 34790 edges'
G_test = nx.empty_graph(G.number_of_nodes())
#'Graph with 1930 nodes and 0 edges'

test_ratio = 0.3
n_links = G_train.number_of_edges() #34790
n_links_test = math.ceil(test_ratio * n_links) #3479

selected_links_id = np.random.choice(np.arange(n_links), size=n_links_test, replace=False)
#(3479,)

network_adj_matrix = nx.adjacency_matrix(G) #Returns adjacency matrix of G
network_adj_matrix = scipy.sparse.triu(network_adj_matrix, k=1) 
#returns upper triangular portion of a matrix in sparse format
row_index, col_index = network_adj_matrix.nonzero() 
#returns indices of elements that are non-zero
links = [(x, y) for x, y in zip(row_index, col_index)]

selected_links = []
for link_id in selected_links_id:
    selected_links.append(links[link_id])
G_train.remove_edges_from(selected_links)
G_test.add_edges_from(selected_links)

G_train.number_of_edges(), G_test.number_of_edges()
#(23758, 10183)

# ## Sampling negative links
k = 1
n_links_train_pos = G_train.number_of_edges()
n_links_test_pos = G_test.number_of_edges()
n_links_train_neg = k * n_links_train_pos
n_links_test_neg = k * n_links_test_pos

neg_network = nx.empty_graph(G.number_of_nodes())
links_neg = list(nx.non_edges(G))
neg_network.add_edges_from(links_neg)

n_links_neg = neg_network.number_of_edges()
n_links_neg #1826695

selected_links_neg_id = np.random.choice(np.arange(n_links_neg), size=n_links_train_neg + n_links_test_neg, replace=False)

neg_network_train = nx.empty_graph(G.number_of_nodes())
neg_network_test = nx.empty_graph(G.number_of_nodes())

selected_links = []
for i in range(n_links_train_neg):
    link_id = selected_links_neg_id[i]
    selected_links.append(links_neg[link_id])
neg_network_train.add_edges_from(selected_links)

selected_links = []
for i in range(n_links_train_neg, n_links_train_neg + n_links_test_neg):
    link_id = selected_links_neg_id[i]
    selected_links.append(links_neg[link_id])
neg_network_test.add_edges_from(selected_links)

neg_network_train.number_of_nodes(), neg_network_test.number_of_nodes()
#(1930, 1930)
neg_network_train.number_of_edges(), neg_network_test.number_of_edges()
# (23758, 10183)

# ## Grouping training and test links

all_links_train = list(G_train.edges) + list(neg_network_train.edges) #62622
label_train = [1] * len(G_train.edges) + [0] * len(neg_network_train.edges) #62622

all_links_test = list(G_test.edges) + list(neg_network_test.edges) #6958
label_test = [1] * len(G_test.edges) + [0] * len(neg_network_test.edges) #6958

y_train, y_test = np.array(label_train), np.array(label_test)
#62622, 6958

# ## Extracting enclosing subgraph for each links

link = all_links_train[12]
link #(1123, 276)

fringe = [link] # [(1123, 276)]
subgraph = nx.Graph()

def enclosing_subgraph(fringe, network, subgraph, distance):
    neighbor_links = []
    for link in fringe:
        u = link[0]
        v = link[1]
        neighbor_links = neighbor_links + list(network.edges(u))
        neighbor_links = neighbor_links + list(network.edges(v))
    tmp_subgraph = subgraph.copy()
    tmp_subgraph.add_edges_from(neighbor_links)
    # Remove duplicate and existed edge
    neighbor_links = [li for li in tmp_subgraph.edges() if li not in subgraph.edges()]
    tmp_subgraph = subgraph.copy()
    tmp_subgraph.add_edges_from(neighbor_links, distance=distance, inverse_distance=1/distance)
    return neighbor_links, tmp_subgraph


fringe, subgraph = enclosing_subgraph(fringe, G_train, subgraph, distance=1)
#subgraph - 'Graph with 34 nodes and 61 edges'
nx.draw(subgraph, with_labels=True)


def extract_enclosing_subgraph(link, network, size=10):
    fringe = [link]
    subgraph = nx.Graph()
    distance = 0
    subgraph.add_edge(link[0], link[1], distance=distance)
    while subgraph.number_of_nodes() < size and len(fringe) > 0:
        distance += 1
        fringe, subgraph = enclosing_subgraph(fringe, network, subgraph, distance)
    
    tmp_subgraph = network.subgraph(subgraph.nodes)
    additional_edges = [li for li in tmp_subgraph.edges if li not in subgraph.edges]
    subgraph.add_edges_from(additional_edges, distance=distance+1, inverse_distance=1/(distance+1))
    return subgraph

e_subgraph = extract_enclosing_subgraph(link, G_train)
#'Graph with 34 nodes and 482 edges'
nx.draw(e_subgraph, with_labels=True)

#e_subgraph[6]
#AtlasView({5: {'distance': 1, 'inverse_distance': 1.0}, 7: {'distance': 1, 'inverse_distance': 1.0}, 12: {'distance': 2, 'inverse_distance': 0.5}})

#%%timeit
extract_enclosing_subgraph(link, G_train)
#3.18 ms ± 261 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

#get_ipython().run_cell_magic('timeit', '', 'extract_enclosing_subgraph(link, network_train)')
#1.44 ms ± 229 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

#%%timeit
for link in all_links_train:
    e_subgraph = extract_enclosing_subgraph(link, G_train)
#7min 6s ± 1min 56s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    
#get_ipython().run_cell_magic('timeit', '', 'for link in all_links_train:\n    e_subgraph = extract_enclosing_subgraph(link, network_train)')
#23.2 s ± 1.41 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

# ## Subgraph encoding
# ### Palette-WL for vertex ordering
def compute_geometric_mean_distance(subgraph, link):
    u = link[0]
    v = link[1]
    subgraph.remove_edge(u, v)
    
    n_nodes = subgraph.number_of_nodes()
    u_reachable = nx.descendants(subgraph, source=u)
    v_reachable = nx.descendants(subgraph, source=v)
#   print(u_reachable, v_reachable)
    for node in subgraph.nodes:
        distance_to_u = 0
        distance_to_v = 0
        if node != u:
            distance_to_u = nx.shortest_path_length(subgraph, source=node, target=u) if node in u_reachable else 2 ** n_nodes
        if node != v:
            distance_to_v = nx.shortest_path_length(subgraph, source=node, target=v) if node in v_reachable else 2 ** n_nodes
        subgraph.nodes[node]['avg_dist'] = math.sqrt(distance_to_u * distance_to_v)
    
    subgraph.add_edge(u, v, distance=0)
    
    return subgraph

e_subgraph = compute_geometric_mean_distance(e_subgraph, link)
#'Graph with 34 nodes and 482 edges'

avg_dist = nx.get_node_attributes(e_subgraph, 'avg_dist')
avg_dist

def prime(x):
    if x < 2:
        return False
    if x == 2 or x == 3:
        return True
    for i in range(2, x):
        if x % i == 0:
            return False
    return True

prime_numbers = np.array([i for i in range (10000) if prime(i)], dtype=np.int64)
#array([   2,    3,    5, ..., 9949, 9967, 9973], dtype=int64)

prime_numbers.shape
#(1229,)

def palette_wl(subgraph, link):
    tmp_subgraph = subgraph.copy()
    if tmp_subgraph.has_edge(link[0], link[1]):
        tmp_subgraph.remove_edge(link[0], link[1])
    avg_dist = nx.get_node_attributes(tmp_subgraph, 'avg_dist')
    
    df = pd.DataFrame.from_dict(avg_dist, orient='index', columns=['hash_value'])
    df = df.sort_index()
    df['order'] = df['hash_value'].rank(axis=0, method='min').astype(np.int64)
    df['previous_order'] = np.zeros(df.shape[0], dtype=np.int64)
    adj_matrix = nx.adjacency_matrix(tmp_subgraph, nodelist=sorted(tmp_subgraph.nodes)).todense()
    while any(df.order != df.previous_order):
        df['log_prime'] = np.log(prime_numbers[df['order'].values])
        total_log_primes = np.ceil(np.sum(df.log_prime.values))
        df['hash_value'] = adj_matrix * df.log_prime.values.reshape(-1, 1) / total_log_primes + df.order.values.reshape(-1, 1)
        df.previous_order = df.order
        df.order = df.hash_value.rank(axis=0, method='min').astype(np.int64)
    nodelist = df.order.sort_values().index.values
    return nodelist

nodelist = palette_wl(e_subgraph, link)
nodelist #(34)
#array([1123,  276,  520,  354,  964, 1719, 1731,  119, 1093,  667,  935,
#       1145,  984, 1177, 1605, 1621, 1906, 1007, 1415, 1869,  242, 1598,
#        893,  876, 1809, 1313,  944,  180,   42,  394,  226, 1408,  382,
#       1572], dtype=int64)

size = 10
if len(nodelist) > size:
    nodelist = nodelist[:size]
    e_subgraph = e_subgraph.subgraph(nodelist)
    nodelist = palette_wl(e_subgraph, link)

nodelist #(10,)
# array([ 276, 1123, 1719,  667,  520,  964,  354, 1731,  119, 1093],
#      dtype=int64)

nx.draw(e_subgraph, with_labels=True)

#e_subgraph.nodes[7]
#{'avg_dist': 0.0}

# ### Represent enclosing subgraphs as adjacency matrices

def sample(subgraph, nodelist, weight='weight', size=10):
    adj_matrix = nx.adjacency_matrix(subgraph, weight=weight, nodelist=nodelist).todense()
    vector = np.asarray(adj_matrix)[np.triu_indices(len(adj_matrix), k=1)]
    d = size * (size - 1) // 2
    if len(vector) < d:
        vector = np.append(vector, np.zeros(d - len(vector)))
    return vector[1:]

sample(e_subgraph, nodelist, size=10)
#array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
#       1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 #     dtype=int32)

# ### Subgraph encoding test

link #(1123, 276)

e_subgraph = extract_enclosing_subgraph(link, G_train, size=10)
e_subgraph = compute_geometric_mean_distance(e_subgraph, link)
nodelist = palette_wl(e_subgraph, link)
if len(nodelist) > size:
    nodelist = nodelist[:size]
    e_subgraph = e_subgraph.subgraph(nodelist)
    nodelist = palette_wl(e_subgraph, link)
embeded = sample(e_subgraph, nodelist, size=10)

embeded #(44,)
#array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
#       1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#      dtype=int32)

# ## Enclosing subgraph encoding for each links

def encode_link(link, network, weight='weight', size=10):
    e_subgraph = extract_enclosing_subgraph(link, network, size=size)
    e_subgraph = compute_geometric_mean_distance(e_subgraph, link)
    nodelist = palette_wl(e_subgraph, link)
    if len(nodelist) > size:
        nodelist = nodelist[:size]
        e_subgraph = e_subgraph.subgraph(nodelist)
        nodelist = palette_wl(e_subgraph, link)
    embeded_link = sample(e_subgraph, nodelist, weight=weight, size=size)
    return embeded_link

#%%timeit
np.array(list(map(partial(encode_link, network=G_train, weight='inverse_distance', size=10), all_links_train)))
#1min 36s ± 1.99 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

X_train = np.array(list(map(partial(encode_link, network=G_train, weight='weight', size=10), all_links_train)))

X_train.shape #(5739, 44)

X_test = np.array(list(map(partial(encode_link, network=G_train, weight='weight', size=10), all_links_test)))
X_test # (639, 44)

from sklearn.utils import shuffle
X_train_shuffle, y_train_shuffle = shuffle(X_train, y_train)

X_train_shuffle.shape, y_train.shape

# ## Neural Network Learning

model = MLPClassifier(hidden_layer_sizes=(32, 32, 16),
                      alpha=1e-3,
                      batch_size=128,
                      learning_rate_init=0.001,
                      max_iter=100,
                      verbose=True,
                      early_stopping=False,
                      tol=-10000)
model.fit(X_train_shuffle, y_train_shuffle)
predictions = model.predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(label_test, predictions, pos_label=1)
auc = metrics.auc(fpr, tpr)
print("The auc is: ", auc)


