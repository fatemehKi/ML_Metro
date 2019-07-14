"""
Created on Mon May 20 03:38:53 2019
@author: Fatemeh Kiaie
@description: this project implement the handwritten image classification using
 classification models
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, precision_score, classification_report, confusion_matrix, accuracy_score


################ loading dataset
def load_file(file):
    '''loading csv fel to pd dataframe'''
    df=pd.read_csv(file)
    df = df.sample(n=5000) #Due to the size of dataset randomly select 50K sample
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
    '''Principal Component Analysis'''
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
    '''feature elimination using RFE'''
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
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train)      
    parameters = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
    return(model, parameters, ac_s, ac_s_train, cm, result)    
    
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
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train)
    parameters= {'n_neighbors':[5, 10, 50, 100]}        
    return(model, parameters, ac_s, ac_s_train, cm, result)

################ Random Forest 
def RaFo_C(X_train, y_train, X_test, y_test):
    model=RandomForestClassifier()#default Random Forest 
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    y_pred = model.predict(X_test)
    ac_s = accuracy_score(y_test,y_pred) #how many prediction matches with the values 
    cm = confusion_matrix(y_test, y_pred)
    result = classification_report(y_test,y_pred)
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train)
    parameters= { 'n_estimators': [5, 10, 50], 'max_depth' : [5,10,15]}
    return(model, parameters, ac_s, ac_s_train, cm, result)

################ Adaboost
def ada_C(X_train, y_train, X_test, y_test):
    model=AdaBoostClassifier() 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ac_s = accuracy_score(y_test,y_pred) #how many prediction matches with the values 
    cm = confusion_matrix(y_test, y_pred)
    result = classification_report(y_test,y_pred)
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train)   
    parameters = { 'n_estimators': [5, 10, 50] }
    return(model, parameters, ac_s, ac_s_train, cm, result)

################ SVM
def svm_C(X_train, y_train, X_test, y_test):
    model=SVC(kernel='linear') 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ac_s = accuracy_score(y_test,y_pred) #how many prediction matches with the values 
    cm = confusion_matrix(y_test, y_pred)
    result = classification_report(y_test,y_pred)   
    #checking for overfiting
    ac_s_train = model.score(X_train,y_train)    
    parameters = {'C': [1.0,0.1,0.01], 'kernel':['linear','rbf','poly'] }
    return(model, parameters, ac_s, ac_s_train, cm, result)
   

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

    
################ K-fold algorithm.. can be replaced by cross_val_score(estimator)
def k_fld(k, X, y, model):
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
    
    ######################################### Define input dataset
    file_name="A_Z_Handwritten_Data.csv"
    dataset=load_file(file_name)
    
    ######################################### Cleaning using backward filling
    dataset=cleaning(dataset)  
    
    ######################################### Correlation Analysis
    cor_=cor(dataset)
    print(cor_)
    
    ######################################### Normalization/scaling is not required
    
    
    ######################################### last column is considered to be the target
    midx=1 
    X, y=trg(dataset, midx)
    

    ######################################### Feature Elimination
    
    #1. checking the high variance feature
    r=pca_(X)
    
    #2. Feature elimination using RFE
    RFE_adj_R2, support_, ranking_, X_sub = Ftr_elm(X, y)
    
    ######################################### splittig the data-fr is fraction of test size for splitting
    fr=0.2 
    X_train, y_train, X_test, y_test= splt(X,y,fr)    
    
    ######################################### ---model training and performance results--- #########################################
    
    #k value in kfold cross validation
    k=4  
    
    ###################### Logistic Regression
    model_LOR, parameters_LOR, accuracy_score_LOR, accuracy_score_train_LOR, cm_LOR, result_LOR=LoR(X_train, y_train, X_test, y_test, y, X)
    print("Considered model is :", model_LOR)
    print("accuracy score value is :", accuracy_score_LOR)    
    print("accuracy score value for training is :", accuracy_score_train_LOR)
    print("confusion matrix is :", cm_LOR)
    print("Classification Report is :", result_LOR )    
    ###gridsearchCV and kfold
    grd_best_score_LOR, grd_best_params_LOR=grdsrch_cv(X, y, model_LOR, parameters_LOR)
    print('best score coming from grid search CV score: ', grd_best_score_LOR)
    print('best parameters coming from grid search CV score: ', grd_best_params_LOR)
    ##another method of kfold.. 
    kfld_cv_score_LOR= k_fld_cv(k, X, y, model_LOR)
            
    
    ###################### KNN
    model_KNN, parameters_KNN, accuracy_score_KNN, accuracy_score_train_KNN, cm_KNN, result_KNN= KNN_C(X_train, y_train, X_test, y_test)
    print("Considered model is :", model_KNN)
    print("accuracy score value is :", accuracy_score_KNN)     
    print("accuracy score value for training is :", accuracy_score_train_KNN)
    print("confusion matrix is :", cm_KNN)
    print("Classification Report is :", result_KNN )
    ###gridsearchCV and kfold
    grd_best_score_KNN, grd_best_params_KNN=grdsrch_cv(X, y, model_KNN, parameters_KNN)
    print('best score coming from grid search CV score: ', grd_best_score_KNN)
    print('best parameters coming from grid search CV score: ', grd_best_params_KNN)
    ##another method of kfold.. 
    kfld_cv_score_KNN= k_fld_cv(k, X, y, model_KNN)
 
    
    ###################### Random Forest
    model_RF, parameters_RF, accuracy_score_RF, accuracy_score_train_RF, cm_RF, result_RF = RaFo_C(X_train, y_train, X_test, y_test)
    print("Considered model is :", model_RF)
    print("accuracy score value is :", accuracy_score_RF)   
    print("accuracy score value for training is :", accuracy_score_train_RF)
    print("confusion matrix is :", cm_RF)
    print("Classification Report is :", result_RF ) 
    ###gridsearchCV and kfold
    grd_best_score_RF, grd_best_params_RF=grdsrch_cv(X, y, model_RF, parameters_RF)
    print('best score coming from grid search CV score: ', grd_best_score_RF)
    print('best parameters coming from grid search CV score: ', grd_best_params_RF)
    ##another method of kfold.. 
    kfld_cv_score_RF= k_fld_cv(k, X, y, model_RF)
    
    
    ###################### Adaboost  
    model_ADA, parameters_ADA, accuracy_score_ADA, accuracy_score_train_ADA, cm_ADA, result_ADA = ada_C(X_train, y_train, X_test, y_test)
    print("Considered model is :", model_ADA)
    print("accuracy score value is :", accuracy_score_ADA)    
    print("accuracy score value for training is :", accuracy_score_train_ADA)
    print("confusion matrix is :", cm_ADA)
    print("Classification Report is :", result_ADA )   
    ###gridsearchCV and kfold
    grd_best_score_ADA, grd_best_params_ADA=grdsrch_cv(X, y, model_ADA, parameters_ADA)
    print('best score coming from grid search CV score: ', grd_best_score_ADA)
    print('best parameters coming from grid search CV score: ', grd_best_params_ADA)
    ##another method of kfold.. 
    kfld_cv_score_ADA= k_fld_cv(k, X, y, model_ADA)
    
    
    ###################### SVM
    model_SVM, parameters_SVM, accuracy_score_SVM, accuracy_score_train_SVM, cm_SVM, result_SVM = svm_C(X_train, y_train, X_test, y_test)
    print("Considered model is :", model_SVM)
    print("accuracy score value is :", accuracy_score_SVM)
    print("accuracy score value for training is :", accuracy_score_train_SVM)
    print("confusion matrix is :", cm_SVM)
    print("Classification Report is :", result_SVM ) 
    ###gridsearchCV and kfold
    grd_best_score_SVM, grd_best_params_SVM=grdsrch_cv(X, y, model_SVM, parameters_SVM)
    print('best score coming from grid search CV score: ', grd_best_score_SVM)
    print('best parameters coming from grid search CV score: ', grd_best_params_SVM)
    ##another method of kfold.. 
    kfld_cv_score_SVM= k_fld_cv(k, X, y, model_SVM)
  
    
    ######################################### Result Visualization
    plt.rcdefaults()
    objects = ('Logistic_Regression', 'KNN', 'Random_Forest', 'AdaBoost', 'SVM')
    y_pos = np.arange(len(objects))
   
    
    SCORE_DE = [accuracy_score_LOR, accuracy_score_KNN, accuracy_score_RF, accuracy_score_ADA, accuracy_score_SVM]
    plt.bar(y_pos, SCORE_DE, align='center', alpha=0.5, color = 'orange')
    plt.xticks(y_pos, objects)
    plt.ylabel('Score Comparison With Default Parameters')
    plt.title('Performance Evaluation')
    plt.show()
    
    
    
    SCORE_OV = [(accuracy_score_train_LOR-accuracy_score_LOR), (accuracy_score_train_KNN-accuracy_score_KNN), (accuracy_score_train_RF-accuracy_score_RF), (accuracy_score_train_ADA-accuracy_score_ADA),(accuracy_score_train_SVM-accuracy_score_SVM)]
    plt.bar(y_pos, SCORE_OV, align='center', alpha=0.5, color= 'BLUE')
    plt.xticks(y_pos, objects)
    plt.ylabel('Difference Between Train and Test Score')
    plt.title('Performance Evaluation')
    plt.show()
    
    SCORE_kfold = [ kfld_cv_score_LOR,  kfld_cv_score_KNN, kfld_cv_score_RF, kfld_cv_score_ADA, kfld_cv_score_SVM]
    plt.bar(y_pos, SCORE_kfold, align='center', alpha=0.5, color='red' )
    plt.xticks(y_pos, objects)
    plt.ylabel('Cross Validation Score ')
    plt.title('Performance Evaluation')
    plt.show()
    
    
    
    SCORE_BE = [grd_best_score_LOR, grd_best_score_KNN, grd_best_score_RF, grd_best_score_ADA, grd_best_score_SVM]
    plt.bar(y_pos, SCORE_BE, align='center', alpha=0.5, color='darkgreen')
    plt.xticks(y_pos, objects)
    plt.ylabel('Best Achieved Score')
    plt.title('Performance Evaluation')
    plt.show()
    
    
