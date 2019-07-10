# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:45:33 2019

@author: mfatemeh
"""
import pandas as pd
import nltk 
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


####################################cleaning dataset
stop_words = set(stopwords.words('english'))
#lemmatize and stem
ps = PorterStemmer()
lem = WordNetLemmatizer()
dataset=pd.read_table("Restaurant_Reviews.tsv")
corpus=[]

for i in range(len(dataset)):
    text = dataset.Review.iloc[i]
    text = text.lower() #changes evrything lower case
    nopunct_text = re.sub('[^a-z0-9]',' ',text) #remove non alphanumeric characters
    #tokenize
    tokens = WhitespaceTokenizer().tokenize(nopunct_text)
    filtered = [ps.stem(lem.lemmatize(w)) for w in tokens if w not in stop_words]
    filtered_text=' '.join(filtered)
    corpus.append(filtered_text)
 
####################################### change it to the x                                                 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #bag of words tools
bow=CountVectorizer(ngram_range=(1,2))
X=bow.fit_transform(corpus).toarray()
word_list=bow.vocabulary_ #feature arrangements is alphabetric
y=dataset.Liked.values #to change it to numpy

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

#classifying using Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


from sklearn.ensemble import RandomForestClassifier
#rf=RandomForestClassifier(max_depth=3)
rf=RandomForestClassifier(max_depth=25)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test) 


############using TfidfVectorize
tfidf= TfidfVectorizer()
X=tfidf.fit_transform(corpus).toarray()
word_list=tfidf.vocabulary_ #feature arrangements is alphabetric
y=dataset.Liked.values #to change it to numpy


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

#classifying using Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


from sklearn.ensemble import RandomForestClassifier
#rf=RandomForestClassifier(max_depth=3)
rf=RandomForestClassifier(max_depth=25)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test) 


#############Adding new text
text='great plcae super'
bow2=CountVectorizer()
x_test=bow.transform([text]).toarray()# the reason we use just transform is we already made the fit
# for the test we use transform--fit is like how many feature we are going to use
clf.predict(x_test) #predicted one..

text2='worse experienc ever'
bow3=CountVectorizer()
x_test2=bow.transform([text2]).toarray()# the reason we use just transform is we already made the fit
# for the test we use transform--fit is like how many feature we are going to use
clf.predict(x_test2) #predicted one..     

text3='worse best'
bow4=CountVectorizer()
x_test3=bow.transform([text3]).toarray()# the reason we use just transform is we already made the fit
# for the test we use transform--fit is like how many feature we are going to use
clf.predict(x_test3) #predicted one..

                    
