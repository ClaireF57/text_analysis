import pandas as pd
# import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from collections import Counter
import numpy as np
import json
import math
import heapq


ENCODING = 'ISO 8859-1'
NB_KEYWORDS = 5

# ## Text preprocessing
# 1- we prepare, clean and tokenize the text
# 2 - we calculate the tf_idf of each word for each job considering the whole corpus
#
# ... take some time

#https://www.quora.com/How-can-I-tokenize-a-string-without-using-built-in-function

def text_preprocessing(para):
    WORD = re.compile(r'\w+')
    nlp = WORD.findall(para)

    # remove non alphanumeric character inculding punctuation
    nlp = [re.sub('[^A-Za-z]', '', word) for word in nlp]

    # remove stop words ; get each word in lowercase ; stem each word
    stop_words = set(stopwords.words('english'))
    stop_words = list(stop_words) + ['','The' , 'we', 'you'] # add other words to remove
    porter_stemmer = PorterStemmer()
    preprocessed = [porter_stemmer.stem(word.lower()) for word in nlp if not word in stop_words]
    return preprocessed


# ## text to vec using tf_idf
#
# - we already vectorized each job ad but they don't share the same vocabulary
# - we have to be able to compare job ads to each other , so we need a global "dictionnary" for all words in the job ads
# - But there are a lot of words so a lot of features
# - But we are only interested in useful words (neither too common , but frequent enough to bring information that can be used-
# - to do this we are to study frequency and existence of each words and select a threshold

# https://stevenloria.com/tf-idf/

def tf(word, txt_tokenized):
    return txt_tokenized.count(word) / len(txt_tokenized)

def n_containing(word, txt_corpus):
    return sum(1 for txt_tokenized in txt_corpus if word in txt_tokenized)

def idf(word, txt_corpus):
    return math.log(len(txt_corpus) / (1 + n_containing(word, txt_corpus)))


# ## Get keywords

def get_keywords(tfidf_vector_list, nb_keyword, df_vocab_useful):
    C = list(tfidf_vector_list)
    t = [np.mean(x) for x in zip(*C)]
    b = [t.index(number) for number in heapq.nlargest(nb_keyword, t)]
    keywords = list(df_vocab_useful.iloc[b,].index)
    return keywords

# Opens and read the json data file
with open("trainingDataScrapped.json","r", encoding=ENCODING) as file:
    contents = file.read()

ads = pd.read_json(contents)

# For test cases, take only a part of the set -- TODO Remove in production
small_set = pd.DataFrame()
for searchTerm in ads.searchTerm.unique():
    small_set = small_set.append(ads.loc[ads.searchTerm == searchTerm].head(10))
ads = small_set

# Preprocess the text
ads["text_process"] = ads['description'].map(text_preprocessing)
corpus = list(ads["text_process"])

# Take only unique words
corpus_word = []
for wordlist in enumerate(corpus):
    corpus_word +=  wordlist[1]

corpus_word2 = list(set(corpus_word)) # Set takes only unique words

# TF IDF
# TF OK
TF = Counter(corpus_word)
# Why does IDF give the same negative number to all words?
IDF = {word : idf(word, corpus) for word in corpus_word2}

# Transform previous TF and IDF in DataFrames
Term_Freq = pd.DataFrame( {"Word" : list(TF.keys()), "TF" :list(TF.values())} )
Term_IDF = pd.DataFrame( {"Word" : list(IDF.keys()), "IDF" :list(IDF.values())} )

# Merge both based on word
df_vocab = Term_Freq.merge(Term_IDF, on="Word")
df_vocab.TF = df_vocab.TF / sum(df_vocab.TF)

# How precise is it?

# Set filters
minIDF = len(corpus) * (1 / 100)  # at least in 1% of the documents
maxIDF = len(corpus) * (90 / 100)  # maximum in 90% of the documents
minTF = 100 / len(corpus_word) # at least 100 occurences of the words

# Implement the filters
df_vocab_useful = df_vocab.loc[(df_vocab.TF > minTF) & (df_vocab.IDF < math.log(len(corpus) / (1 + minIDF)) )& (df_vocab.IDF > math.log(len(corpus) / (1 + maxIDF)) ),]
df_vocab_useful = df_vocab_useful.set_index('Word')


df_vocab_useful["initial_value"] =  0
vocab = df_vocab_useful.loc[: ,("initial_value")].to_dict()

corpus_word_list = list(df_vocab_useful.index)

# Creates new column and affects the TF_IDF values
ads["tfidf_voc"]  = ads.text_process.map(lambda txt_tokenized :[tf(word, txt_tokenized) * df_vocab_useful.loc[word, "IDF"] for word in corpus_word_list])

keywords_set = []
for searchTerm in ads.searchTerm.unique():
    tfidf_vector_list = ads.loc[ads.searchTerm == searchTerm, "tfidf_voc"]
    keywords_set += get_keywords(tfidf_vector_list, NB_KEYWORDS, df_vocab_useful)

print(keywords_set)
