install.packages("devtools")
library(devtools)
install_github("mkarikom/RSoptSC")
library(RSoptSC)
#data = read.csv('D:\\Masters\\Datasets\\Baron-Human Pancreas\\data1NormWithColNamesRowNames.csv')
#data = read.csv('D:\\Masters\\Datasets\\Baron-Human Pancreas\\data2NormWithColNamesRowNames.csv')

data = read.csv('./data/data2NormWithColNamesRowNames.csv')
row.names(data) = data$X
data = subset(data, select = -c(X))
df = subset(data, select = -c(assigned_cluster))
#df = subset(data, select = -c(cellType))
df = abs(df)
df = t(df)
data_m = data.matrix(df)
S <- SimilarityM(lambda = 0.05, 
                 data = data_m,
                 dims = 3,
                 pre_embed_method = 'tsne',
                 perplexity = 20, 
                 pca_center = TRUE, 
                 pca_scale = TRUE)
low_dim_mapping <- RepresentationMap(similarity_matrix = S$W,
                                     flat_embedding_method = 'tsne',
                                     join_components = TRUE,
                                     perplexity = 35,
                                     theta = 0.5,
                                     normalize = FALSE,
                                     pca = TRUE,
                                     pca_center = TRUE,
                                     pca_scale = TRUE,
                                     dims = 2,
                                     initial_dims = 2)
library(igraph)
adj = low_dim_mapping$adj_matrix
row.names(adj) = colnames(df)
colnames(adj) = colnames(df)
graph = graph_from_adjacency_matrix(adj)
edgelist = get.edgelist(graph)
write.csv(edgelist,"./code/RSoptSC/HumanD2/edgelist_HumanD2.csv")
write.csv(adj,"./code/RSoptSC/HumanD2/adjacencymatrix_HumanD2.csv")
write.csv(colnames(adj),"./code/RSoptSC/HumanD2/cellnames_HumanD2.csv")



