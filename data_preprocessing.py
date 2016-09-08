import scipy
import numpy
import sklearn
import os
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer

path_to_files = "data/NLM_500/documents/"

paths = os.listdir(path_to_files)

print("Test")

data = []
for file in paths:
    if file.endswith("txt"):
       readable = open(path_to_files + file,encoding="ISO-8859-1")
       data.append(readable.read())
from  nltk.stem.snowball import  SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens =[]
    #filter out non alphabetic things
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems



#Define Vectorizer Paramaters

tfidfvectorizer = TfidfVectorizer(max_df=0.9,min_df=0.1,max_features=100000,stop_words='english',tokenizer=tokenize_and_stem,ngram_range=(1,3))


tfidf_matrix = tfidfvectorizer.fit_transform(data)
"Dumping the TFIDF Matrix after Creation"
from sklearn.externals import joblib

joblib.dump(tfidfvectorizer,'tfidfvectorizer.pkl')

joblib.dump(tfidf_matrix,'tfidf_matrix.pkl')

print("The Input data has been pickled into tfidf matrix and vectorizer")




