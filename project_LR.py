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

##### Splitting--assuming the last column is the target
def splt(df, fr):
    df_train = df.sample(frac=fr,random_state=0)
    X_train=df_train.iloc[:,:-1]
    y_train=dataset.iloc[:,-1]
    df_test = df.drop(df_train.index)
    X_test=df_test.iloc[:,:-1]
    y_test=df_test.iloc[:,-1]
    return(X_train, y_train, X_test, y_test)

##### Initilization
#def init:
#    X = tf.constant(np.random.randn(), name = "X")
#    W = tf.constant(np.random.randn(), name = "W")
#    b = tf.constant(np.random.randn(), name = "b")
#    #Y = tf.add(tf.matmul(W, X),b)
#    W = tf.Variable(np.random.randn(), dtype=np.float)
#    b = tf.placeholder(np.random.randn())
#    X = tf.placeholder(tf.float32, shape=[n_x, None])
#    Y = tf.placeholder(tf.float32, shape=[n_y, None])
#
##### Model
#def LR()
#    x=tf.placeholder(tf)
#    logits = tf.matmul(x, weights) + biases
    
   
if __name__ == '__main__':  
    
    #define input dataset
    file_name="OnlineNewsPopularity.csv"
    dataset=load_file(file_name)
    dataset=cleaning(dataset)
    dataset=rmv_usls(dataset, 'url')
    cor(dataset)
    fr=0.8 #fraction of splitting
    X_train, y_train, X_test, y_test= splt(dataset,fr)
    n_x = X_train.shape
    eta =tf.constant(0.01, name='learning_rate')
    stp = tf.constant(1000, name='no_step')
    dis_stp = tf.constant(100, name='display_step')
    
    
    W = tf.Variable(np.random.randn(), dtype=tf.float32)
    b = tf.Variable(np.random.randn(), dtype=tf.float32)
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    y_act = tf.placeholder(tf.float32)
    
    LR= W*X+b
    Loss=tf.reduce_sum(LR-y_act)    
    optimizer=tf.train.Gradient    
    
    
    
    
    
    
    
    

# Linear regression (Wx + b)
def linear_regression(inputs):
    return inputs * W + b


# Mean square error
def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / (2 * n_samples)


# SGD Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Compute gradients
grad = tfe.implicit_gradients(mean_square_fn)

# Initial cost, before optimizing
print("Initial cost= {:.9f}".format(
    mean_square_fn(linear_regression, train_X, train_Y)),
    "W=", W.numpy(), "b=", b.numpy())

# Training
for step in range(num_steps):

    optimizer.apply_gradients(grad(linear_regression, train_X, train_Y))

    if (step + 1) % display_step == 0 or step == 0:
        print("Epoch:", '%04d' % (step + 1), "cost=",
              "{:.9f}".format(mean_square_fn(linear_regression, train_X, train_Y)),
              "W=", W.numpy(), "b=", b.numpy())
    
