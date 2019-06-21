# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:01:52 2019

@author: gsaikia
"""

#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
X = pd.read_csv('Mall_Customers.csv')

#import clustering model
from sklearn.cluster import KMeans

#train model
model = KMeans(n_clusters=2,random_state=0)
model.fit(X)

#find the final cluster assignments
y_pred = model.predict(X)


#Visualising the clusters for two-dimensional datapoints
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')





