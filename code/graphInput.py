# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 17:01:53 2021

@author: sheen
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scipy.io as sio

#Human pancreas dataset
#df = pd.read_csv('D:/Masters/My_thesis/Code/RSoptSC/HumanD3/edgelist_HumanD3.csv')
df = pd.read_csv('./RSoptSC/HumanD3/edgelist_HumanD3.csv')
df.drop(['Unnamed: 0'], axis=1, inplace = True)

# creating instance of labelencoder
labelencoder = LabelEncoder()

df1 = pd.DataFrame()
# Assigning numerical values and storing in another column
df1['v1'] = labelencoder.fit_transform(df['V1'])
df1['v2'] = labelencoder.fit_transform(df['V2'])

df1.to_csv(r'./data/HumanD3/edgelist_encoded_HumanD3.edgelist',index=None, sep=' ', mode='a', header = None)
df1.to_csv(r'./data/HumanD3/edgelist_encoded_HumanD3.txt',index=None, sep=' ', mode='a', header = None)
df1.to_csv(r'./data/HumanD3/edgelist_encoded_HumanD3.csv',index=None, mode='a')


