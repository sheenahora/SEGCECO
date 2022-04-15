# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:19:47 2020

@author: Akram
"""
import numpy as np
# data preprocessing
import os
#os.chdir("D:/Academic/Fall2020/PatternRecognition/Project/humanPancreas/GSE84133-Baron")

import pandas as pd
import scanpy as sc
data1 = pd.read_csv("./data/rawdata/GSM2230757_human1_umifm_counts.csv" ,index_col=0)
type(data1)
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()

#results_file = 'D:/Academic/Fall2020/PatternRecognition/Project/humanPancreas/GSE84133-Baron/results/GSE84133h1.h5ad'# the file that will store the analysis results

results_file = './Results/GSE84133h1.h5ad'# the file that will store the analysis results

# type(data1)
# cols1 = data1.iloc[1, 2:]
# type(cols1)

cols = data1.columns
genelist =[i for i in cols if (i != 'Unnamed: 0') and ( i !='barcode') and (i !='assigned_cluster')]
# type(genelist)
# genelist[0:3]
numberOfGenes = len(genelist)
df= pd.read_csv("./data/rawdata/GSM2230757_human1_umifm_counts.csv", usecols =[i for i in cols if (i != 'Unnamed: 0')],index_col=0)#and ( i !='barcode') and (i != 'assigned_cluster')
type(df)
dfNotEncode = df
# Notice that 'assigned_cluster' contains string names to represent cell types. Let's encode them to integer
print("Before encoding: ")
len(np.unique(dfNotEncode.iloc[0:430,0]))
cellTypes = dfNotEncode.iloc[0:430,0]
cellTypes.to_csv("./cellTypes.csv")
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# df = dfNotencode
df.assigned_cluster = pd.factorize(df.assigned_cluster)[0]

# df.assigned_cluster = le.fit_transform(df.assigned_cluster)

np.unique(df.assigned_cluster)

print("\nAfter encoding: ")
# print(df.iloc[0:430,0])
len(np.unique(df.iloc[0:430,0]))
cellTypesEncoded = df.iloc[0:430,0]
cellTypesEncoded.to_csv("./cellTypesEncoded.csv")

#########################
df.to_csv("./data1.csv")

def dataSetAnalysis(df):
    #view starting values of data set
    print("Dataset Head")
    print(df.head(3))
    print("=" * 30)
    
    # View features in data set
    print("Dataset Features")
    print(df.columns.values)
    print("=" * 30)
    
    # View How many samples and how many missing values for each feature
    print("Dataset Features Details")
    print(df.info())
    print("=" * 30)
    
    # view distribution of numerical features across the data set
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 30)
    
    # view distribution of categorical features across the data set
    # print("Dataset Categorical Features")
    # print(df.describe(include=['O']))
    # print("=" * 30)

dataSetAnalysis(df)

# d = pd.read_csv("./data1.csv")
# d.head()

# Scanpy ********************************************************************
adata = sc.read_csv("./data1.csv", first_column_names=True)
adata.obs_names
adata.var_names
adata.X
# adata.write_csvs("./anndata1.csv",skip_data=False, sep=',')
# Observation names are not unique. To make them unique, call `.obs_names_make_unique`.
adata # n_obs × n_vars = 1937 × 20126
# adataNonUniqueObs = adata
# len(np.unique(adataNonUniqueObs.obs_names))
adata.obs_names_make_unique()
# adata #adata # n_obs × n_vars = 1937 × 20126!!!
# len(np.unique(adata.obs_names))
# adata.var_names_make_unique() 
sc.pl.highest_expr_genes(adata, n_top=20, )
# # Basic filtering.
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# # With pp.calculate_qc_metrics, we can compute many metrics very efficiently.
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# # A violin plot of some of the computed quality measures:

# # the number of genes expressed in the count matrix
# # the total counts per cell
# # the percentage of counts in mitochondrial genes

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

# # Remove cells that have too many mitochondrial genes expressed or too many total counts.
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# # Actually do the filtering by slicing the AnnData object.
adata_filtered = adata[adata.obs.n_genes_by_counts < 4000, :]
# adata_filtered.X
adata = adata_filtered
labelsBeforeNorm = adata.X[:,0]
# Total-count normalize (library-size correct) the data matrix X to 30,000 reads per cell, 
# so that counts become comparable among cells.
# adata.raw = adata
# rd = adata.raw[:, 1:].X
# rd = adata.raw[:, 1:].var
# adata.chunk_X([10])
# adata[:,1:]
sc.pp.normalize_total(adata, target_sum=3e4)#['X'[:,1:]] #*********************If choosing target_sum=1e6, this is CPM normalization.
max(adata.X[:,0])
# Logarithmize the data.
sc.pp.log1p(adata)#**********************
max(adata.X[:,0])

adata.X[:,0] = labelsBeforeNorm
max(adata.X[:,0])
sc.pl.highest_expr_genes(adata, n_top=20, )

# *********************************************************
# Identify highly-variable genes. (feature selection) 
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=.5)
sc.pl.highly_variable_genes(adata)
sc.pl.highly_variable_genes(adata[:,1:])
max(adata.X[:,0])
# Set the .raw attribute of AnnData object to the normalized and logarithmized raw gene expression 
# for later use in differential testing and visualizations of gene expression. 
# This simply freezes the state of the AnnData object.

adata.raw = adata
# adata.raw[:, 1:].X
# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
# sc.pp.scale(adata, max_value=10)
# max(adata.X[:,0])
# adata.X[:,0] = labelsBeforeNorm
# max(adata.X[:,0])
# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed. Scale the data to unit variance.
# Regress out (mostly) unwanted sources of variation.
# a simple batch correction method is available via pp.regress_out()
# sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
# max(adata.X[:,0])
# adata.X[:,0] = labelsBeforeNorm
# max(adata.X[:,0])


adataWithCluster = adata
adata.var.highly_variable.assigned_cluster = True

adata = adata[:, adata.var.highly_variable]
# adata = adataWithCluster

# Join col(gene) names to X values
rowNames = adata.obs_names

type(rowNames)
rows = pd.DataFrame(index=adata.obs_names)
type(rows)
np.savetxt("barcodes1.csv", rows)

hvgList = adata.var_names
n_hvg = len(adata.var_names)
hvg = pd.DataFrame(index=adata.var_names)
hvgString = ','.join(hvgList)
# Save adata as a csv file with row and col names
np.savetxt("data1NormWithColNames.csv", adata.X, delimiter=",", header = hvgString, comments='')

dataNorm = pd.read_csv('./data1NormWithColNames.csv')
dataSetAnalysis(dataNorm)
dataNorm.set_index(rowNames, inplace=True)
dataSetAnalysis(dataNorm)

# np.savetxt("data1NormWithColNamesRowNames.csv", dataNorm,delimiter=",", )
dataNorm.to_csv("./data/data1NormWithColNamesRowNames.csv")
dataNormRows = pd.read_csv('./data/data1NormWithColNamesRowNames.csv')
dataSetAnalysis(dataNormRows)
#########################
