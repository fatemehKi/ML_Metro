# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:40:02 2019

@author: mfatemeh
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##we don't have y because it is the unsupervised
X=pd.read_csv("Mall_Customers.csv").values #either use iloc or the value thing


from sklearn.cluster import KMeans


wcss = []
for i in range(1,11):
    model = KMeans(n_clusters=i,random_state=0)
    model.fit(X)
    wcss.append(model.inertia_)
    
plt.plot(range(1,11),wcss)
plt.xlabel("K")
plt.ylabel("WCSS")
plt.title("Finding optimal K")


#import clustering model
from sklearn.cluster import KMeans

#train model
model = KMeans(n_clusters=6,random_state=0)
model.fit(X)

#find the final cluster assignments
y_pred = model.predict(X)

#Visualising the clusters for two-dimensional datapoints
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'brown', label = 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'orange', label = 'Cluster 5')
plt.scatter(X[y_pred == 5, 0], X[y_pred == 5, 1], s = 100, c = 'pink', label = 'Cluster 6')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
