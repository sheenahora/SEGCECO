# SEGCECO: Subgraph Embedding of Gene expression matrix for prediction of CEll-cell COmmunication 

**Introduction**

In this work, we present a novel method, SEGCECO, using an attributed graph convolutional neural network approach for predicting cell-cell communication from single-cell RNA seq data. SEGCECO capture the latent and explicit attributes of undirected, attributed graphs constructed from gene expression profile of individual cells. Our proposed method consists of three main steps: 
1) Preprocessing step 
2) Cell-cell communication network (CCN) creation 
3) Applying Graph convolutional network technique 
 
Before applying machine learning techniques, the primary step is to preprocess the data for downstream analysis. Once the data is preprocessed, CCN is constructed using SoptSC method. The last module of our pipeline is our method, SEGCECO, which takes processed sc-RNA seq data and CCN to create an attributed graph dataset and gives the prediction output.

**Pipeline of the proposed framework**

[Pipeline](https://github.com/sheenahora/SEGCECO/blob/master/Overview.png)

**Pre-processing step**

To run pre-processing steps in single-cell RNA sequencing dataset, execute the **preprocessing.py** from the code folder in project home directory.

**Cell-cell communication network (CCN) creation**

To create the CCN from pre-processed single-cell RNA sequencing data, execute the **CCN.R** from the project code folder in home directory. The output of this step is edgelist of the graph which is passed to **graphInput.py** to perform the label encoding and get the output in the below format:

1 2\
1 3\
1 4\
...

**SEGCECO module**

This module consists of 2 steps:

1) Gene selection from pooling layer: The pooling layer in our proposed method consists of selecting genes (with the threshold of 300) by Information Gain feature selection method. Execute the **Features.py** from the project code folder in home directory. The input to this step is preprocessed single-cell RNA sequencing dataset and the output of this step is **attributes_IG.csv** in the data folder respective to each dataset.

2) Run the SEGCECO module - Execute **SEGCECO/Main_LinkPredict.py** from the project code folder in home directory. The input to this step is **edgelist_encoded_HumanD1.txt** (CCN) and  **attributes_IG_HumanD1.csv** (explicit attributes) in the data folder respective to each dataset.

**Version Requirements for SEGCECO module**

1) python 3.5.5
2) networkx 2.0
3) tensorflow 1.7.0
4) numpy 1.14.2

**Dataset Used**

Single-cell RNA sequencing dataset
| Dataset | Tissue |  Accession | #Cells |  #Genes
| --- | --- | --- | --- | --- | 
| Baron-human1 | Human-Pancreas | GSM2230757 | 1,937 | 20,125
| Baron-human2 | Human-Pancreas | GSM2230758 | 1,724 | 20,125
| Baron-human3 | Human-Pancreas | GSM2230759 | 3,605 | 20,125
| Baron-human4 | Human-Pancreas | GSM2230760 | 1,303 | 20,125
| Baron-mouse1 | Mouse-Pancreas | GSM2230761 |   822 | 14,878
| Baron-mouse2 | Mouse-Pancreas | GSM2230762 | 1,064 | 14,878

**Comparison Study**

To evaluate the performance of SEGCECO, we compared it with latent feature methods (i.e. Node2vec, LINE, DeepWalk, SpectralClustering, GAE, VGAE) and state-of-the-art method, WLNM. The code for each methods can be found in **Embedding_Methods** folder of the project code folder in home directory. The steps to generate embeddings are cited in code/Embedding_Methods/Generate embeddings.txt. The node embedding methods (i.e. Node2vec, LINE, DeepWalk, SpectralClustering) gives the feature representation of nodes in a network as node embedding. These node embedding outputs are stored in **Embedding_Results** folder in the code folder in home directory. Thus, an additional step (**EdgeFeatures.py**) is required to learn edge features from node embeddings in order to predict links as a binary classification problem. 

**Acknowledgements**

I would like to express my gratitude to my supervisor, Dr. Luis Rueda, for his assistance and encouragement, Akram Vasighizaker, a PhD student for her collaboration on this project, and the University of Windsor Office of Research and Innovation.

**References**

1) Code for "Zhang, Muhan, and Yixin Chen. Weisfeiler-lehman neural machine for link prediction. KDD 2017": https://github.com/KienMN/Weisfeiler-Lehman-Neural-Machine
2) Code for "Grover, Aditya, and Jure Leskovec. node2vec: Scalable feature learning for networks. KDD 2016.": https://github.com/aditya-grover/node2vec
3) Code for "Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. KDD 2014.": https://github.com/phanein/deepwalk
4) SoptSC R package: https://mkarikom.github.io/RSoptSC
5) Code for "Zhang, Muhan, and Yixin Chen. Link prediction based on graph neural networks. Advances in neural information processing systems 31 (2018).": https://github.com/XuSShuai/SEAL-for-link-prediction
6) Code for "Tang, Jian, et al. Line: Large-scale information network embedding. Proceedings of the 24th international conference on world wide web. 2015.": https://github.com/tangjianpku/LINE
7) Code for "Kipf, Thomas N., and Max Welling. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308 (2016).": https://github.com/tkipf/gae
8) Code for Spectral Clustering: https://github.com/lucashu1/link-prediction
