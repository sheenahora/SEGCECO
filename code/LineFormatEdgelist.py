# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:04:13 2022

@author: sheen
"""

import networkx as nx
import os

os.getcwd()
#os.chdir("D:/Masters/My_thesis/Code")
#os.chdir("D:/Masters/My_thesis/Code/Datasets/HumanD3")

G = nx.read_edgelist('./data/HumanD3/edgelist_encoded_HumanD3.txt', nodetype=int, create_using=nx.Graph())
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1
G = G.to_undirected()
print(nx.info(G))

nx.write_edgelist(G, "./data/HumanD3/Weighted_edgelist_encoded_HumanD3.edgelist", delimiter=' ', data=['weight'])
