# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:52:29 2019

@author: mfatemeh
"""

"""
Created on Mon July 8 03:38:53 2019
@author: Fatemeh Kiaie
@description: this project implement the lego price prediction using regression
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

################ loading dataset
def load_file(file):
    '''loading csv fel to pd dataframe'''
    df=pd.read_csv(file)
    return df

################ Data Cleaning
def cleaning(df):
    '''handling missing values usig backward filling'''
    print(df.isnull().sum())    
    if (df.isnull().sum().any() !=0):
        df=df.dropna(how='all')
        df=df.fillna(method='bfill', inplace=True)
    return df

################ Dropping unrelated features
def rmv_usls(df, col):
    '''droping features that are not related'''
    df=df.drop(col, axis=1)
    return df

################ Correlation pre analysis and heatmap
def cor(df):
    '''Correlation of features analysis'''
    cor = df.corr(method='pearson')
    sns.heatmap(cor,annot=True)
    return(cor)

################ Target Feature seperation
def trg(df, midx):
    '''target identification'''
    X=df.iloc[:,:-midx].values             
    y=df.iloc[:,-midx].values
    return X,y

################ Scaling
def scl(X,y):
    '''scaling'''
    sc_X = StandardScaler()
    X_s=sc_X.fit_transform(X)
    sc_y=StandardScaler()
    y_s=sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))
    return(X_s,y_s,sc_X, sc_y)

################ PCA
def pca_(X):
    '''Principal '''
    pca=PCA()
    pcs=pca.fit_transform(X)
    ratio=pca.explained_variance_ratio_
    if (max(ratio)-min(ratio)>0.5):
        print('pca can remove some features')
    else:
        print('pca did not help')
    return(ratio)
    
################ Feature elimination
def Ftr_elm(X, y):
    ''' feature elimination using RFE'''
    adj_R2 = []
    feature_set = []
    max_adj_R2_so_far = 0
    n = len(X)
    k = len(X[0])
    selected_ranking=[]
    for i in range(1,k+1):
        selector = RFE(LinearRegression(), i,verbose=1)
        selector = selector.fit(X, y)
        current_R2 = selector.score(X,y)
        current_adj_R2 = 1-(n-1)*(1-current_R2)/(n-i-1) 
        adj_R2.append(current_adj_R2)
        feature_set.append(selector.support_)
        if max_adj_R2_so_far < current_adj_R2:
            max_adj_R2_so_far = current_adj_R2
            selected_features = selector.support_
            #selected_ranking= selector.ranking_
            selected_ranking.append(selector.ranking_)
        print('End of iteration no. {}'.format(i))
        print('selector support is :', selector.support_)
        #print('selected ranking is ;', selector.ranking_)
        print('selected ranking is ;', selected_ranking)      
    X_sub = X[:,selected_features]    
    return (adj_R2, selector.support_, selector.ranking_, X_sub )

################ Splitting
def splt(X, y, fr):
    '''spliting dataset to the train, test sets for the input and target'''
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=fr,random_state=0)
    return(X_train, y_train, X_test, y_test)

################ Linear Regression
def LiR(X_train, y_train, X_test, y_test, X_s, y_s, sc_y):
    '''Linear Regression model and the scores'''
    model = LinearRegression()
    #train the model
    model.fit(X_train, y_train)    
    y_pred = model.predict(X_test)    
    #scoring the linear regression model
    #score_= model.score(X_test, y_test) 
    score_ = model.score(X_train, y_train) 
    #kfold cross validation--first method:      
    kfold_score= cross_val_score(model, X_s, y_s, cv=4)
    n = len(X_s)
    k = len(X_s[0])
    adj_R2_ = 1-(n-1)*(1-score_)/(n-k-1)     
    y_pred = sc_y.inverse_transform(y_pred.reshape(len(y_pred),1)).reshape(len(y_pred))
    y_test = sc_y.inverse_transform(y_test.reshape(len(y_test),1)).reshape(len(y_test))
    MSR_= mean_squared_error(y_test, y_pred)
    X_modified = sm.add_constant(X_train)
    lin_reg = sm.OLS(y_train,X_modified)
    result = lin_reg.fit()    
    #gridsearch parameter
    parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    return(model, parameters, score_, adj_R2_, MSR_, result, kfold_score)    
    
################ KNN 
def KNN_R(X_train, y_train, X_test, y_test):
    #sc=list()
    #for k in range(1, 21):
    model=neighbors.KNeighborsRegressor(n_neighbors = k)
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
       # sc.append(model.score(X_test, y_test))
    sc=model.score(X_test, y_test)
    parameters= {'n_neighbors':[2,3,4,5,6,7,8,9]}        
    return (model, parameters, sc) 

################ Random Forest 
def RaFo_R(X_train, y_train, X_test, y_test):
    model=RandomForestRegressor(n_estimators=15, max_depth=2, random_state=0)#random forest with the 15 trees 
    model.fit(X_train, y_train)
    #rf.score(X_train, y_train)
    sc=model.score(X_test, y_test)
    parameters= { 'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy']}
    return(model, parameters, sc )

################ Adaboost
def ada_R(X_train, y_train, X_test, y_test):
    model=AdaBoostRegressor(n_estimators=15, random_state=0) 
    model.fit(X_train, y_train)
    #rf.score(X_train, y_train)
    sc=model.score(X_test, y_test)
    parameters = { 'n_estimators': [50, 100], 'learning_rate' : [0.01,0.05,0.1,0.3,1],  'loss' : ['linear', 'square', 'exponential'] }
    return(model, parameters, sc )

################ SVM
def svm_R(X_train, y_train, X_test, y_test):
    model=SVR(kernel='linear') 
    model.fit(X_train, y_train)
    #rf.score(X_train, y_train)
    sc=model.score(X_test, y_test)
    parameters = {'C': [1.0,0.1,0.001], 'kernel':['linear','rbf','poly'] ,'degree':[2,3] }
    return(model, parameters, sc )
   

################ Grid search
def grdsrch_cv(X_s, y_s, model, parameters):
    ''' Grid search scoring results'''
    grid = GridSearchCV(model,parameters, cv=4) ## second kfold method: 4 fold cross valiation
    grid.fit(X_s, y_s)
    return(grid.best_score_)
    
################ K-fold algorithm.. third method..first method is using cross_val_score(estimator) in the model(used in LiR), 2nd is gread_search
def k_fld(k, X_s, model):
    ''' k fold algorithm implementation'''
    scores = []
    max_score = 0
    kf = KFold(n_splits=k,random_state=0,shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_s[train_index], X_s[test_index]
        y_train, y_test = y_s[train_index], y_s[test_index]
        model.fit(X_train,y_train)
        #see performance score in k
        current_score = model.score(X_test,y_test)
        scores.append(current_score)
        if max_score < current_score:
            max_score = current_score
            #best_model = model
    return(max_score) 



##################################################################################
#--------------------------------------- Main ----------------------------------#
if __name__ == '__main__':  
    
    #define input dataset
    file_name="OnlineNewsPopularity.csv"
    dataset=load_file(file_name)
    
    dataset=cleaning(dataset)
    dataset=rmv_usls(dataset, 'url')    
    cor_=cor(dataset)
    print(cor_)
    
    #last column is considered to be the 
    midx=1 
    X, y=trg(dataset, midx)
    
    #normalization/scaling is required
    X_s,y_s,sc_X, sc_y=scl(X,y)
    
    #checking the high variance feature
    r=pca_(X_s)
    
    #Feature eliination using RFE
    RFE_adj_R2, support_, ranking_, X_sub = Ftr_elm(X_s, y_s)
    
    #splittig the data-fr is fraction of test size for splitting
    fr=0.2 
    X_train, y_train, X_test, y_test= splt(X_s,y_s,fr)
    
    ###model training and performance results
    k=4    
    ##Linear Regression
    model, parameters, score_, adj_R2_, MSR_, result, kfold_score=LiR(X_train, y_train, X_test, y_test, X_s, y_s, sc_y)
    print("Considered model is :", model)
    print("score value is :", score_)
    print("adjusent R^2 value is :", adj_R2_)
    print("Mean square error is :", MSR_)
    print('P value is:', result.summary())    
    ###gridsearchCV and kfold, 2nd and 3rd kfold metho.. 2nd is inside gridsearchcv and third is using the function "kfld"
    grd_best_score_LiR=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_LiR, ' in model: ', model)
    #kfold implementation: third method    
    kfld_max_score_LiR= k_fld(k, X_s, model)
        
    ##KNN
    model, parameters, sc_knn= KNN_R(X_train, y_train, X_test, y_test)
    print('accuracy score for default KNN with equal to: ', sc_knn)
    ###gridsearchCV and kfold
    grd_best_score_kNN=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_kNN, ' in model: ', model)
    kfld_max_score_KNN= k_fld(k, X_train, X_test, X_s, model)
    
    ##Random Forest
    model, parameters, sc_RF = RaFo_R(X_train, y_train, X_test, y_test)
    print('accuracy score for Random Forest equals to: ', sc_RF)
    ###gridsearchCV and kfold
    grd_best_score_RaFo_R=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_RaFo_R, ' in model: ', model)
    kfld_max_score_RaFo_R= k_fld(k, X_train, X_test, X_s, model)
    
    ##Adaboost    
    model, parameters, sc_ada = ada_R(X_train, y_train, X_test, y_test)
    print('accuracy score for Adaboost equals to: ', sc_ada)    
    ###gridsearchCV and kfold
    grd_best_score_ada_R=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_ada_R, ' in model: ', model)
    kfld_max_score_ada_R= k_fld(k, X_train, X_test, X_s, model)
    
    ##SVM
    model, parameters, sc_svm = svm_R(X_train, y_train, X_test, y_test)
    print('accuracy score for SVM equals to: ', sc_svm)
    ###gridsearchCV and kfold
    grd_best_score_svm_R=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_svm_R, ' in model: ', model)
    kfld_max_score_svm_R= k_fld(k, X_train, X_test, X_s, model)
  
    
    
    
    
