# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:04:07 2021

@author: sheen
"""

import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest, f_classif

#attributes_mat = pd.read_csv("D:/Masters/Datasets/Baron-Human Pancreas/data1NormWithColNamesRowNames.csv", index_col=0)
#attributes_mat = pd.read_csv("D:/Masters/Datasets/Baron-Human Pancreas/data2NormWithColNamesRowNames.csv", index_col=0)

#attributes_mat = pd.read_csv("D:/Masters/Datasets/Baron-Human Pancreas/HumanD3NormWithColNamesRowNames.csv", index_col=0) #(2638, 1838)

attributes_mat = pd.read_csv("./data/HumanD3NormWithColNamesRowNames.csv", index_col=0)

X = attributes_mat.iloc[:, attributes_mat.columns != 'assigned_cluster'].values
#X = np.absolute(X)
X
y = attributes_mat.iloc[:,attributes_mat.columns == 'assigned_cluster'].values.ravel()
y

#X_kbestfeatures = SelectKBest(f_classif, k = 300).fit_transform(X, y)
#X_kbestfeatures.shape

bestfeatures_ig = SelectKBest(mutual_info_classif, k= 300)
fit_ig = bestfeatures_ig.fit(X,y)
df_scores_ig = pd.DataFrame(fit_ig.scores_)

data1 = attributes_mat
data1.drop(columns='assigned_cluster', inplace = True)
data1.head(10)

df_columns = pd.DataFrame(data1.columns)

# concatenate dataframes
feature_scores_ig = pd.concat([df_columns, df_scores_ig],axis=1)
feature_scores_ig.columns = ['Feature_Name','Score']  # name output columns

#Printing top 20 features IG
print(feature_scores_ig.nlargest(20,'Score')) 
# export selected features to .csv
df_feat_ig = feature_scores_ig.nlargest(20,'Score')
df_feat_ig.to_csv('./data/HumanD3/20_feature_selected_ig.csv', index=False)

#Printing top 300 features IG
print(feature_scores_ig.nlargest(300,'Score')) 
# export selected features to .csv
df_feat_ig = feature_scores_ig.nlargest(300,'Score')
df_feat_ig.to_csv('./data/HumanD3/300_feature_selected_ig.csv', index=False)

df_feat_ig['Feature_Name']
attributes = pd.read_csv("./data/HumanD3NormWithColNamesRowNames.csv",  usecols =[i for i in df_feat_ig['Feature_Name']])
attributes.to_csv("./data/HumanD3/attributes_IG_HumanD3.csv", index=False)


