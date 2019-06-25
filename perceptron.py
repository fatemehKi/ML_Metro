# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:16:00 2019

@author: mfatemeh
"""

import numpy as np
from sklearn.datasets import load_iris
### we use only the length column in order to have the easy plot
import matplotlib.pyplot as plt


dataset= load_iris()
#extracting the setosa and versicolor datapointd datapoint using s
X=dataset.data[:100, [0,2]] #100 rows with first and third column
y=dataset.target[:100]

#importing the 
from sklearn.linear_model import Perceptron
clf=Perceptron(n_iter=10)

clf.fit(X,y)
clf.score(X,y) ### it is the score of the training score that's why it is 100% 

# visualizing the decision boundry of the trained perceptron
w0=clf.intercept_[0]
w1=clf.coef_[0,0]
w2=clf.coef_[0,1]
#w0+w1x1+w2x2=0  
#to plot the lines, we need to select two random lines and use it in the plot
#we find the minimum of x1 and maximum of x1 to have a long lines
#we find the min and max in x1 to plot.. x1 is the first column.. to expand the figure
x1=np.array([X[:,0].min(), X[:,0].max()])
x2=-w1*x1/w2-w0/w2


#Visulaising the data points
plt.scatter(X[y==0,0], X[y==0,1], c='blue')
plt.scatter(X[y==1,0], X[y==1,1], c= 'red')
plt.plot(x1,x2,c='green')

