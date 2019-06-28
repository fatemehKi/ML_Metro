# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:52:35 2019

@author: mfatemeh
"""

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

dataset= load_digits()

X=dataset.data
y=dataset.target

#no need to scale and there is not missing value
#there is no missing value
#there is no need for encoding.. no categorical data


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=10)

acc_sc=list()

for k in range(1, 21):
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    acc_sc.append(accuracy_score(y_test, y_pred))
    confusion_matrix(y_test,y_pred)
    
acc_sc

plt.plot(range(1, 21), acc_sc)
plt.xlabel("K")
plt.ylabel("Accuracy")

#####################second method.. using the whole data set
scores=[]
for i in range(1,21):
    clf=KNeighborsClassifier(n_neighbors=i)
    scores.append(cross_val_score(clf,X, y, cv=4).mean()) ## using 4fold method
    
import matplotlib.pyplot as plt
plt.plot(range(1, 21), scores)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title('Finding optimal K')  ## one is the best
        
         
