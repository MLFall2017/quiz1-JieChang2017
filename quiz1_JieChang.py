#get package
import numpy as np
from numpy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
import random
import pandas as pd
from pandas import DataFrame, read_csv
from sklearn.decomposition import PCA
import csv

#Problem 1
#soln: use numpy
#1: read data into matrix, ignoring the header
df_2 = np.genfromtxt('dataset_1.csv', delimiter=',', dtype=float)[1:]
print('Dataset_1:')
print(df_2)

#2:center the data
row_mean = np.mean(df_2, axis=0)#column means
#print(row_mean)
df_2_c = df_2 - row_mean
#print(df_2_c)

#3:transpose the matrix to get cov matrix
df_3_c = df_2_c.transpose()
cov_df_3_c = np.cov(df_3_c)
print('Covariance Matrix for Dataset_1:')
print(cov_df_3_c)
#4:get eigenvalue and eigenvector
eiva,eive = LA.eig(cov_df_3_c)
print('Eigenvalues for Dataset_1:')
print(eiva)
print('Correspoding Eigenvectors for Dataset_1:')
print(eive)#column vectors are eigenvectors

#5:sort eigenpairs in descending order
#1):make a list of (eigenvalue,eigenvector) tuples
eig_pairs = [(np.abs(eiva[i]), eive[:,i])
             for i in range(len(eiva))]
#2):sort the tupels from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
#3):print the sorted dicreasing eigenvalues to check
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
#6:find projection matrix
list_a = []
for i in range(len(eig_pairs)):
    list_a.append(eig_pairs[i][1])
pro_matrix = np.array(list_a).T
print('Projection Matrix:')
print(pro_matrix)

#7:calculate pc
pc = np.matmul(df_2_c, pro_matrix)
pc_12 = pc[:,0:2]#get the first two projection clm vectors
print('PCA for dataset_1:')
print(pc_12)




#Problem 3
#(2)use linalg to find eigenvalues and eigenvectors
#create matrix
a = np.array([[0,-1],[2,3]], dtype=float)
print(a)

#find eigenpairs
lam,p = LA.eig(a)
print('eigenvalues for a')
print(lam)
print('eigenvectors for a')
print(p)
