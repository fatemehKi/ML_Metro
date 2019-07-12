"""
Created on Mon May 20 03:38:53 2019
@author: Fatemeh Kiaie
@description: this project implement the lego price prediction using linear 
regression
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score, classification_report, confusion_matrix, accuracy_score


################ loading dataset
def load_file(file):
    '''loading csv fel to pd dataframe'''
    df=pd.read_csv(file)
    df = df.sample(n=1000) #Due to the size of dataset randomly select 50K sample
    return df

################ Data Cleaning
def cleaning(df):
    '''handling missing values using backward filling'''
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
    #print(df.corr(spearman))
    cor = df.corr(method='pearson')
    sns.heatmap(cor,annot=True)
    return(cor)

################ Target Feature seperation
def trg(df, midx):
    '''target identification'''
    X=df.iloc[:,:-midx].values             
    y=df.iloc[:,-midx].values
    return X,y

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
        selector = RFE(LogisticRegression(), i,verbose=1)
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

################ Logistic Regression
def LoR(X_train, y_train, X_test, y_test, y, X):
    '''Linear Regression model and the scores'''
    model = LogisticRegression()
    model.fit(X_train, y_train)    
    y_pred = model.predict(X_test)  
    ac_s = accuracy_score(y_test,y_pred) 
    cm = confusion_matrix(y_test, y_pred)
    result = classification_report(y_test,y_pred)
    parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    return(model, parameters, ac_s, cm, result)    
    
################ KNN 
def KNN_C(X_train, y_train, X_test, y_test):
    ''' KNN regression model'''
    model=KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    y_pred = model.predict(X_test)
    ac_s = accuracy_score(y_test,y_pred) #how many prediction matches with the values 
    cm = confusion_matrix(y_test, y_pred)
    result = classification_report(y_test,y_pred)
    parameters= {'n_neighbors':[10, 50]}        
    return(model, parameters, ac_s, cm, result)

################ Random Forest 
def RaFo_C(X_train, y_train, X_test, y_test):
    model=RandomForestClassifier(n_estimators=15, max_depth=2, random_state=0)#random forest with the 15 trees 
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    y_pred = model.predict(X_test)
    ac_s = accuracy_score(y_test,y_pred) #how many prediction matches with the values 
    cm = confusion_matrix(y_test, y_pred)
    result = classification_report(y_test,y_pred)
    parameters= { 'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy']}
    return(model, parameters, ac_s, cm, result)

################ Adaboost
def ada_C(X_train, y_train, X_test, y_test):
    model=AdaBoostClassifier(n_estimators=15, random_state=0) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ac_s = accuracy_score(y_test,y_pred) #how many prediction matches with the values 
    cm = confusion_matrix(y_test, y_pred)
    result = classification_report(y_test,y_pred)    
    parameters = { 'n_estimators': [50, 100], 'learning_rate' : [0.01,0.05,0.1,0.3,1],  'loss' : ['linear', 'square', 'exponential'] }
    return(model, parameters, ac_s, cm, result)

################ SVM
def svm_C(X_train, y_train, X_test, y_test):
    model=SVC(kernel='linear') 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ac_s = accuracy_score(y_test,y_pred) #how many prediction matches with the values 
    cm = confusion_matrix(y_test, y_pred)
    result = classification_report(y_test,y_pred)
    parameters = {'C': [1.0,0.1,0.001], 'kernel':['linear','rbf','poly'] ,'degree':[2,3] }
    return(model, parameters, ac_s, cm, result)
   

################ Grid search
def grdsrch_cv(X_train, y_train, model, parameters):
    ''' Grid search scoring results'''
    grid = GridSearchCV(model,parameters, cv=None)
    grid.fit(X_train, y_train)
    return(grid.best_score_)
    
################ K-fold algorithm.. can e replaced by cross_val_score(estimator)
def k_fld(k, X_train, X_test, X, model):
    '''k fold function implementation'''
    scores = []
    max_score = 0
    kf = KFold(n_splits=k,random_state=0,shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
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
    file_name="A_Z_Handwritten_Data.csv"
    dataset=load_file(file_name)
    
    #cleaning using backward filling
    dataset=cleaning(dataset)  
    
    #last column is considered to be the target
    midx=1 
    X, y=trg(dataset, midx)
    
    #normalization/scaling is not required
    
    
    #checking the high variance feature
    r=pca_(X)
    
    #Feature elimination using RFE
    RFE_adj_R2, support_, ranking_, X_sub = Ftr_elm(X, y)
    
    #splittig the data-fr is fraction of test size for splitting
    fr=0.2 
    X_train, y_train, X_test, y_test= splt(X,y,fr)    
    
    ###_------------------ model training and performance results -------------
    k=4    
    ###################### Logistic Regression
    model, parameters, accuracy_score_, confusion_matrix, result=LoR(X_train, y_train, X_test, y_test, y, X)
    print("Considered model is :", model)
    print("accuracy score value is :", accuracy_score_)
    print("confusion matrix is :", confusion_matrix)
    print("Classification Report is :", result )    
    ###gridsearchCV and kfold
    grd_best_score_LoR=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_LoR, ' in model: ', model)
    kfld_max_score_LiR= k_fld(k, X_train, X_test, X, model)
        
    
    ###################### KNN
    model, parameters, accuracy_score_, confusion_matrix, result= KNN_C(X_train, y_train, X_test, y_test)
    print("Considered model is :", model)
    print("accuracy score value is :", accuracy_score_)
    print("confusion matrix is :", confusion_matrix)
    print("Classification Report is :", result )
    ###gridsearchCV and kfold
    grd_best_score_kNN=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_kNN, ' in model: ', model)
    kfld_max_score_KNN= k_fld(k, X_train, X_test, X, model)
    
    
    ###################### Random Forest
    model, parameters, accuracy_score_, confusion_matrix, result = RaFo_C(X_train, y_train, X_test, y_test)
    print("Considered model is :", model)
    print("accuracy score value is :", accuracy_score_)
    print("confusion matrix is :")
    print(confusion_matrix)
    print("Classification Report is :", result )    
    ###gridsearchCV and kfold
    grd_best_score_RaFo_C=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_RaFo_C, ' in model: ', model)
    kfld_max_score_RaFo_C= k_fld(k, X_train, X_test, X, model)
    
    
    ###################### Adaboost  
    model, parameters, accuracy_score_, confusion_matrix, result = ada_C(X_train, y_train, X_test, y_test)
    print("Considered model is :", model)
    print("accuracy score value is :", accuracy_score_)
    print("confusion matrix is :")
    print(confusion_matrix)
    print("Classification Report is :", result )     
    ###gridsearchCV and kfold
    grd_best_score_ada_C=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_ada_C, ' in model: ', model)
    kfld_max_score_ada_C= k_fld(k, X_train, X_test, X, model)
    
    
    ###################### SVM
    model, parameters, accuracy_score_, confusion_matrix, result = svm_C(X_train, y_train, X_test, y_test)
    print("Considered model is :", model)
    print("accuracy score value is :", accuracy_score_)
    print("confusion matrix is :")
    print(confusion_matrix)
    print("Classification Report is :", result ) 
    ###gridsearchCV and kfold
    grd_best_score_svm_C=grdsrch_cv(X_train, y_train, model, parameters)
    print('best score coming from grid search CV score: ', grd_best_score_svm_C, ' in model: ', model)
    kfld_max_score_svm_C= k_fld(k, X_train, X_test, X, model)
  
    
    
    
    
