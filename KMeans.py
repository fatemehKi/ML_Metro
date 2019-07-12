# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:01:26 2019

@author: mfatemeh
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##we don't have y because it is the unsupervised
X=pd.read_csv("Mall_Customers.csv").values #either use iloc or the value thing

#import clustering model
from sklearn.cluster import KMeans

#train model
model=KMeans(n_clusters=2,random_state=0) #the default number of k(n_cluster) is 8 and we selected only two clusters
model.fit(X)
#find the final cluster assignment
y_pred=model.predict(X) # we can see the clssification of the output

#Visualising the clusters for two-dimensional datapoints
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

                    
###############################with the 3 clusters
#train model
model=KMeans(n_clusters=3) #the default number of k is 8 and we selected only two clusters
model.fit(X)
#find the final cluster assignment
y_pred=model.predict(X)

#Visualising the clusters for two-dimensional datapoints
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
