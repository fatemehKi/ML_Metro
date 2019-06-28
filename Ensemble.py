# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:19:04 2019

@author: mfatemeh
"""

import numpy as np
import pandas as pd

dataset=pd.read_csv('data_banknote_authentication.txt', header=None, names=['variance', 'skewness', 'kurtosis', 'entropy', 'anthnticity'])


###import models
from sklearn.tree import DecisionTreeClassifier
X=dataset.iloc[:, :-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y)


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


simple=DecisionTreeClassifier(max_depth=1) #simple decision tree
rf=RandomForestClassifier(n_estimators=15, max_depth=3, random_state=0) #random forest with the 15 trees 
ada=AdaBoostClassifier(n_estimators=15, random_state=0) #by default deep is 1.. we can use the other estimator for base_estimator
###adaboost is more general and can be used on non-tree models by using base_estimator parameter

simple.fit(X_train, y_train)
rf.fit(X_train, y_train)
ada.fit(X_train, y_train)

simple.score(X_train, y_train)
rf.score(X_train, y_train)
ada.score(X_train, y_train)

simple.score(X_test, y_test)
rf.score(X_test, y_test)
ada.score(X_test, y_test)
