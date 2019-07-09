# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:26:38 2019

@author: gsaikia
"""

import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "This is a block of text. I'm writing a piece to explain the usage of nltk packages."

text = text.lower() #changes evrything lower case
nopunct_text = re.sub('[^a-z0-9]',' ',text) #remove non alphanumeric characters

#tokenize
tokens = WhitespaceTokenizer().tokenize(nopunct_text)

#remove stopwords
stop_words = set(stopwords.words('english'))

filtered_tokens = []
for token in tokens:
    if token not in stop_words:
        filtered_tokens.append(token)
        
#lemmatize and stem
ps = PorterStemmer()
lem = WordNetLemmatizer()

stemmed_tokens = []
for token in filtered_tokens:
    stemmed_tokens.append(ps.stem(lem.lemmatize(token)))


#equivalent to line 24-36
filtered = [ps.stem(lem.lemmatize(w)) for w in tokens if w not in stop_words]        



        
