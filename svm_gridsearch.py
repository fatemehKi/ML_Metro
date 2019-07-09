# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:42:59 2019

@author: gsaikia
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
dataset = load_digits() 

X = dataset.data
y = dataset.target

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X) 

from sklearn.decomposition import PCA
pca = PCA()
pcs = pca.fit_transform(X)
ratio = pca.explained_variance_ratio_

from sklearn.svm import SVC
clf = SVC()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

from sklearn.model_selection import GridSearchCV
param_dict = {
                'C': [1.0,0.1,0.001], 
                'kernel':['linear','rbf','poly'] ,
                'degree':[2,3]
             }

model = GridSearchCV(clf,param_grid=param_dict,cv=4) 
model.fit(X,y)
model.best_params_
train_scores = model.cv_results_['mean_train_score']
test_scores = model.cv_results_['mean_test_score']
model.best_score_

