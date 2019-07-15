# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:01:59 2019

@author: fatemeh Kiaie
@Title: This file is the final exam of the machine learning
"""

import pandas as pd
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


####################################cleaning dataset
stop_words = set(stopwords.words('english'))
#lemmatize and stem
ps = PorterStemmer()
lem = WordNetLemmatizer()
dataset=pd.read_table("sms_spam_ham.tsv", header=None, names=['label_y', 'sms'])

corpus=[]

for i in range(len(dataset)):
    text = dataset.sms.iloc[i]
    text = text.lower() #changes evrything lower case
    nopunct_text = re.sub('[^a-z0-9]',' ',text) #remove non alphanumeric characters
    #tokenize
    tokens = WhitespaceTokenizer().tokenize(nopunct_text)
    filtered = [ps.stem(lem.lemmatize(w)) for w in tokens if w not in stop_words]
    filtered_text=' '.join(filtered)
    corpus.append(filtered_text)
 


############using TfidfVectorize
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer()
X=tfidf.fit_transform(corpus).toarray()
word_list=tfidf.vocabulary_ 
y=dataset.label_y.values 


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
param_dict = {
                'n_estimators': [5, 10, 15], 
                'max_depth' : [4,5,6,7,8],
                'criterion' :['gini', 'entropy']
             }


model = GridSearchCV(rf,param_grid=param_dict, cv=4) 
model.fit(X,y)
model.best_params_
model.best_score_
