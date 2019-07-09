# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:49:42 2019

@author: mfatemeh
"""

import nltk 
nltk.download()

from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer

text="this is a block of text. I am writing a piece to explain the use of nlp packages."
text='Feet wolves talked cats'

######tokenize
tokenizer1=WhitespaceTokenizer()#extract based o white space
tokenizer2=WordPunctTokenizer()#extract based on the white space as well as punctuation
tokenizer3=TreebankWordTokenizer()

tokens1=tokenizer1.tokenize(text)
tokens2=tokenizer2.tokenize(text)
tokens3=tokenizer3.tokenize(text)

######
#best is first try to lemmetizing and then stem
from nltk.stem import PorterStemmer, WordNetLemmatizer

ps=PorterStemmer()
lem=WordNetLemmatizer()

lemmatized_tokens=[]
for token in tokens3:
    lemmatized_tokens.append(lem.lemmatize(token))

#lemmatized and stemmed
lemmatized_tokens=[]
for token in tokens3:
    lemmatized_tokens.append(ps.stem(lem.lemmatize(token)))
    
#filtering out the stop words
from nltk.corpus import stopwords
stop_words=stopwords.words('english')



stoped_rmv=[]
for w in lemmatized_tokens:
    for s in stop_words:
        if (w!=s):
            stoped_rmv.append(w)

#### correct method        
filtered_tokens=[]
for token in lemmatized_tokens:
    if token not in stop_words:
        filtered_tokens.append(token)
################################################################
#replacing the punctuation 
import re
text="this is a block of text. I am writing a piece to explain the use of nlp packages."
##first lower case all words
text=text.lower()
nopunct_text= re.sub('[^a-z0-9]',' ',text) #you need to write "for what character" "what pattern" on "which text" we need to 
### ^ means everything except.. means not sign

###tokenize
token =WhitespaceTokenizer().tokenize(nopunct_text)

##remove stopwords
stop_words=stopwords.words('english')

filtered_tokens=[]
for token in lemmatized_tokens:
    if token not in stop_words:
        filtered_tokens.append(token)
        
##list comprehensive method below:
filtered_tokens = [w for w in token if w not in stop_words]

ps= PorterStemmer()
lem=WordNetLemmatizer()

stemmed_tokens=[]
for token in filtered_tokens:
    stemmed_tokens.append(ps.stem(lem.lemmatize(token)))
####list comprehensive method of the loop above from line 73 till the end
filtered=[ps.stem(lem.lemmatize(w)) for w in token if w not in stop_words]
