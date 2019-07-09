# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:22:28 2019

@author: Kian
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt



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
    df=df.drop(columns=col)
    return df

#### Correlation pre analysis and heatmap
def cor(df):
    '''Correlation of features analysis'''
    print(df.corr())
    cor = df.corr()
    sns.heatmap(cor,annot=True)

##### Normalization
def splt(df):
    X=df[]
    X_train = df.sample(frac=0.8,random_state=0)
    X_test = df.drop(X_train.index)


   
if __name__ == '__main__':  
    
    #define input dataset
    file_name="OnlineNewsPopularity.csv"
    dataset=load_file(file_name)
    dataset=cleaning(dataset)
    dataset=rmv_usls(dataset, 'url')
    cor(dataset)
    
    
