# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:45:17 2019

@author: mfatemeh
"""

import numpy as np

from sklearn.datasets import load_digits
dataset=load_digits()
X=dataset.data
y=dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier



ovsr=OneVsRestClassifier(Perceptron())
ovso=OneVsOneClassifier(Perceptron())

ovsr.fit(X_train, y_train)
ovsr.score(X_test, y_test)

ovso.fit(X_train, y_train)
ovso.score(X_test, y_test)
