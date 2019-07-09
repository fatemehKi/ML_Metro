# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:22:28 2019

@author: Kian
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

##### loading dataset
def load_file(file):
    '''loading csv fel to pd dataframe'''
    df=pd.read_csv(file)
    return df

##### Data Cleaning
def cleaning(df):
    '''handling missing values and invalid data'''
    print(df.isnull().sum())    
    if (df.isnull().sum().any() !=0):
        df=df.dropna(how='all')
        df=df.fillna(method='bfill', inplace=True)
    return df

#### Dropping unrelated features
def rmv_usls(df, col):
    '''droping features that are not related'''
    df=df.drop(col, axis=1)
    return df

#### Correlation pre analysis and heatmap
def cor(df):
    '''Correlation of features analysis'''
    print(df.corr())
    cor = df.corr()
    sns.heatmap(cor,annot=True)

##### Target Feature seperation
def trg(df, midx):
    '''target identification'''
    X=df.iloc[:,:-midx].values             
    y=df.iloc[:,-midx].values
    return X,y


##### Scaling
def scl(X,y):
    '''scaling'''
    sc_X = StandardScaler()
    X=sc_X.fit_transform(X)
    sc_y=StandardScaler()
    sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))
    return(X,y,sc_X, sc_y)

def pca_(X):
    pca=PCA()
    pcs=pca.fit_transform(X)
    ratio=pca.explained_variance_ratio_
    if (max(ratio)-min(ratio)>0.5):
        print('pca can remove some features')
    else:
        print('pca did not help')
    return(ratio)
    

##### Splitting
def splt(X, y, fr):
    '''spliting dataset to the train, test sets for the input and target'''
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=fr,random_state=0)
    return(X_train, y_train, X_test, y_test)

#### Linear Regression
def LR(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    #train the model
    model.fit(X_train, y_train)
    #see the performance of the model 
    y_pred=model.score(X_test, y_test)
    return y_pred
    
####
def inv_scl(sc_X, sc_y, y_pred, y_test):   
    y_pred=sc_y.inverse_transform(y_pred.reshape(len(y_pred),1)).reshape(len(y_pred))
    y_test=sc_y.inverse_transform(y_test.reshape(len(y_test),1)).reshape(len(y_test))
    return(y_pred, y_test)

####
def err(y_test, y_pred):
    return(mean_squared_error(y_test, y_pred))
    
   
if __name__ == '__main__':  
    
    #define input dataset
    file_name="OnlineNewsPopularity.csv"
    dataset=load_file(file_name)
    
    dataset=cleaning(dataset)
    dataset=rmv_usls(dataset, 'url')    
    cor(dataset)
    
    #last column is considered to be the 
    midx=1 
    X, y=trg(dataset, midx)
    
    #normalization/scaling is required
    X, y, sc_X, sc_y=scl(X,y)
    
    #checking the high variance feature
    r=pca_(X)
    
    #splittig the data
    ##fraction of test size for splitting
    fr=0.2 
    X_train, y_train, X_test, y_test= splt(X,y,fr)
    
    #
    y_pred=LR(X_train, y_train, X_test, y_test)
    
    y_pred, y_test= inv_scl(sc_X, sc_y, y_pred, y_test)
    
    meansqr=err(y_test, y_pred)
    

    
    
    
    
    
    
