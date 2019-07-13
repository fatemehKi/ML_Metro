"""
Created on Mon July 8 03:38:53 2019
@author: Fatemeh Kiaie
@description: this project implements regression models to predict nline news popularity
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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
    X=sc_X.fit_transform(X)
    sc_y=StandardScaler()
    y=sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))
    return(X,y,sc_X, sc_y)

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
    score_ = model.score(X_test, y_test) 
    n = len(X)
    k = len(X[0])
    adj_R2_ = 1-(n-1)*(1-score_)/(n-k-1)     
    y_pred = sc_y.inverse_transform(y_pred.reshape(len(y_pred),1)).reshape(len(y_pred))
    y_test = sc_y.inverse_transform(y_test.reshape(len(y_test),1)).reshape(len(y_test))
    MSR_= mean_squared_error(y_test, y_pred)
    X_modified = sm.add_constant(X_train)
    lin_reg_OLS = sm.OLS(y_train,X_modified)
    result = lin_reg_OLS.fit()   
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train)
    parameters = {'fit_intercept':[True,False], 'normalize':[True,False]}
    return(model, parameters, score_, ac_s_train, adj_R2_, MSR_, result)    
    
################ KNN 
def KNN_R(X_train, y_train, X_test, y_test):
    ''' KNN regression model'''
    model=neighbors.KNeighborsRegressor()
    model.fit(X_train, y_train)   
    sc=model.score(X_test, y_test)
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train) 
    parameters= {'n_neighbors':[5, 10, 50, 100]}         
    return (model, parameters, sc, ac_s_train) 

################ Random Forest 
def RaFo_R(X_train, y_train, X_test, y_test):
    '''Random Forest regression model'''
    model=RandomForestRegressor(n_estimators=15, max_depth=2, random_state=0)#random forest with the 15 trees 
    model.fit(X_train, y_train)
    sc=model.score(X_test, y_test)    
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train)
    parameters= { 'n_estimators': [5, 10, 50], 'max_depth' : [5,10,15]}
    return(model, parameters, sc, ac_s_train )

################ Adaboost
def ada_R(X_train, y_train, X_test, y_test):
    '''Adaboost regression model'''
    model=AdaBoostRegressor(n_estimators=15, random_state=0) 
    model.fit(X_train, y_train)
    sc=model.score(X_test, y_test)    
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train) 
    parameters = { 'n_estimators': [5, 10, 50] }
    return(model, parameters, sc, ac_s_train )

################ SVM
def svm_R(X_train, y_train, X_test, y_test):
    '''SVR regression model'''
    model=SVR(kernel='linear') 
    model.fit(X_train, y_train)
    #rf.score(X_train, y_train)
    sc=model.score(X_test, y_test)    
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train) 
    parameters = {'C': [1.0,0.1,0.01], 'kernel':['linear','rbf','poly'] }
    return(model, parameters, sc, ac_s_train)
   

################ Grid search
def grdsrch_cv(X, y, model, parameters):
    ''' Grid search scoring results'''
    grid = GridSearchCV(model,parameters, cv=4)
    grid.fit(X, y)
    return(grid.best_score_, grid.best_params_)
    

################ K-fold cross_val_score(estimator)
def k_fld_cv(k, X, y, model):
    '''k fold cross validation function implementation'''     
    kfold_score= cross_val_score(model, X, y, cv=k).mean()
    return(kfold_score) 

################ K-Fold method
def k_fld(k, X_train, X_test, X, model):
    '''kfold method'''
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
    
    ######################################### Define input dataset
    file_name="OnlineNewsPopularity.csv"
    dataset=load_file(file_name)
    
    ######################################### Cleaning using backward filling
    dataset=cleaning(dataset)
    
    ######################################### Dropping Unrelated column
    dataset=rmv_usls(dataset, 'url')    
    
    ######################################## Correlation Analysis
    cor_=cor(dataset)
    print(cor_)
    
    #last column is considered to be the target.. 
    midx=1 
    X, y=trg(dataset, midx)
    
    #normalization/scaling is required
    X_s, y, sc_X, sc_y=scl(X,y)
    
    #checking the high variance feature
    r=pca_(X_s)
    
    #Feature eliination using RFE
    RFE_adj_R2, support_, ranking_, X_s = Ftr_elm(X_s, y)
    
    #splittig the data-fr is fraction of test size for splitting
    fr=0.2 
    X_train, y_train, X_test, y_test= splt(X_s,y,fr)
    
    ###_------------------ model training and performance results -------------
    k=4    
    ###################### Linear Regression
    #model, parameters, score_, adj_R2_, MSR_, result=LiR(X_train, y_train, X_test, y_test, sc_y, X)
    model_LiR, parameters_LiR, score_LiR, score_train_LiR, adj_R2_LiR , MSR_LiR, result_LiR=LiR(X_train, y_train, X_test, y_test, X_s, y, sc_y)
    print("Considered model is :", model_LiR)
    print("score value is :", score_LiR)    
    print("score value for training is :", score_train_LiR)
    print("adjusent R^2 value is :", adj_R2_LiR)
    print("Mean square error is :", MSR_LiR)
    print('P value is:', result_LiR.summary())    
    ###gridsearchCV and kfold
    grd_best_score_LiR, grd_best_params_LiR=grdsrch_cv(X_s, y, model_LiR, parameters_LiR)
    print('best score coming from grid search CV score: ', grd_best_score_LiR)
    print('best parameters coming from grid search CV score: ', grd_best_params_LiR)
    kfld_cv_score_LiR= k_fld_cv(k, X, y, model_LiR)
        
    ###################### KNN
    model_KNN, parameters_KNN, score_KNN, score_train_KNN= KNN_R(X_train, y_train, X_test, y_test)
    print("Considered model is :", model_KNN)
    print("accuracy score value is :", score_KNN)     
    print("accuracy score value for training is :", score_train_KNN)
    ###gridsearchCV and kfold
    grd_best_score_KNN, grd_best_params_KNN=grdsrch_cv(X_s, y, model_KNN, parameters_KNN)
    print('best score coming from grid search CV score: ', grd_best_score_KNN)
    print('best parameters coming from grid search CV score: ', grd_best_params_KNN)
    kfld_cv_score_KNN= k_fld_cv(k, X, y, model_KNN)
    
    ###################### Random Forest
    model_RF, parameters_RF, score_RF, score_train_RF= RaFo_R(X_train, y_train, X_test, y_test)
    print("Considered model is :", model_RF)
    print("accuracy score value is :", score_RF)     
    print("accuracy score value for training is :", score_train_RF)
    ###gridsearchCV and kfold
    grd_best_score_RF, grd_best_params_RF=grdsrch_cv(X_s, y, model_RF, parameters_RF)
    print('best score coming from grid search CV score: ', grd_best_score_RF)
    print('best parameters coming from grid search CV score: ', grd_best_params_RF)
    kfld_cv_score_RF= k_fld_cv(k, X, y, model_RF)
    
    ###################### Adaboost    
    model_ADA, parameters_ADA, score_ADA, score_train_ADA = ada_R(X_train, y_train, X_test, y_test)
    print("Considered model is :", model_ADA)
    print("accuracy score value is :", score_ADA)     
    print("accuracy score value for training is :", score_train_ADA)
    ###gridsearchCV and kfold
    grd_best_score_ADA, grd_best_params_ADA=grdsrch_cv(X_s, y, model_ADA, parameters_ADA)
    print('best score coming from grid search CV score: ', grd_best_score_ADA)
    print('best parameters coming from grid search CV score: ', grd_best_params_ADA)
    kfld_cv_score_ADA= k_fld_cv(k, X, y, model_ADA)
    
    ###################### SVM
    model_SVM, parameters_SVM, score_SVM, score_train_SVM = svm_R(X_train, y_train, X_test, y_test)
    print("Considered model is :", model_SVM)
    print("accuracy score value is :", score_SVM)     
    print("accuracy score value for training is :", score_train_SVM)
    ###gridsearchCV and kfold
    grd_best_score_SVM, grd_best_params_SVM=grdsrch_cv(X_s, y, model_SVM, parameters_SVM)
    print('best score coming from grid search CV score: ', grd_best_score_SVM)
    print('best parameters coming from grid search CV score: ', grd_best_params_SVM)
    kfld_cv_score_SVM= k_fld_cv(k, X, y, model_SVM)
  
    
    ######################################### Result Visualization
    plt.rcdefaults()
    objects = ('Logistic_Regression', 'KNN', 'Random_Forest', 'AdaBoost', 'SVM')
    y_pos = np.arange(len(objects))
   
    
    SCORE_DE = [score_LiR, score_KNN, score_RF, score_ADA, score_SVM]
    plt.bar(y_pos, SCORE_DE, align='center', alpha=0.5, color = 'darkgreen')
    plt.xticks(y_pos, objects)
    plt.ylabel('Score Comparison With Default Parameters')
    plt.title('Performance Evaluation')
    plt.show()
    
    
    
    SCORE_OV = [(score_train_LiR-score_LiR), (score_train_KNN-score_KNN), (score_train_RF-score_RF), (score_train_ADA-score_ADA),(score_train_SVM-score_SVM)]
    plt.bar(y_pos, SCORE_OV, align='center', alpha=0.5, color= 'BLUE')
    plt.xticks(y_pos, objects)
    plt.ylabel('Difference Between Train and Test Score')
    plt.title('Performance Evaluation')
    plt.show()
    
    SCORE_kfold = [ kfld_cv_score_LiR,  kfld_cv_score_KNN, kfld_cv_score_RF, kfld_cv_score_ADA, kfld_cv_score_SVM]
    plt.bar(y_pos, SCORE_kfold, align='center', alpha=0.5, color='red' )
    plt.xticks(y_pos, objects)
    plt.ylabel('Score With 4Fold')
    plt.title('Performance Evaluation')
    plt.show()
    
    
    
    SCORE_BE = [grd_best_score_LiR, grd_best_score_KNN, grd_best_score_RF, grd_best_score_ADA, grd_best_score_SVM]
    plt.bar(y_pos, SCORE_BE, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Best Achieved Score')
    plt.title('Performance Evaluation')
    plt.show()
    
    
    
