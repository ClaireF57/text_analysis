
# coding: utf-8

# In[6]:


#https://stevenloria.com/tf-idf/

import math


def tf(word, txt_tokenized):
    return txt_tokenized.count(word) / len(txt_tokenized)

def n_containing(word, txt_corpus):
    return sum(1 for txt_tokenized in txt_corpus if word in txt_tokenized)

def idf(word, txt_corpus):
    return math.log(len(txt_corpus) / (1 + n_containing(word, txt_corpus)))

def tfidf(word, txt_tokenized, txt_corpus):
    return tf(word, txt_tokenized) * idf(word, txt_corpus)




# In[51]:


#https://www.quora.com/How-can-I-tokenize-a-string-without-using-built-in-function

def text_preprocessing(para):
    
    #strip the text into sentences
    #nlp=sent_tokenize(para)
    #split the sentences into words: tokenization
    #nlp=[word_tokenize(sent) for sent in nlp]
    
    WORD = re.compile(r'\w+')
    nlp = WORD.findall(para)
    # remove non alphanumeric character , inculding punctuation 
    nlp=[re.sub('[^A-Za-z]', '', word) for word in nlp]
    # remove stop words ; get each work in lowercase ; stem each word
    stop_words=set(stopwords.words("english"))
    stop_words=list(stop_words)+['','The' , "we"] # add other words to remove
    ps = PorterStemmer()
    nlp= [ps.stem(word.lower()) for word in nlp if not word in stop_words]
    return nlp


# In[258]:


import pandas as pd
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from collections import Counter
import numpy as np


# In[10]:


job = pd.read_pickle(open("eng_jobs.p","rb"))


# In[11]:


job.shape


# In[53]:


job["text_process"] = job.text.map(text_preprocessing)
corpus = list(job["text_process"])
job["tfidf_voc"] = job.text_process.map(lambda txt_tokenized :{word : tfidf(word, txt_tokenized, txt_corpus) for word in txt_tokenized})
job["vocabulary_len"] = job.tfidf_voc.map(len)


# In[253]:


job.head()


# In[129]:


corpus_word = []
for wordlist in enumerate(corpus):
    corpus_word +=  wordlist[1]
    
corpus_word2 = list(set(corpus_word))
TF = Counter(corpus_word)
IDF = {word : idf(word, corpus) for word in corpus_word2}
Term_Freq = pd.DataFrame( {"Word" : list(TF.keys()),"TF" :list(TF.values())} )
Term_IDF = pd.DataFrame( {"Word" : list(IDF.keys()),"IDF" :list(IDF.values())} )
df_vocab = Term_Freq.merge(Term_IDF, on="Word")
df_vocab.TF = df_vocab.TF/sum(df_vocab.TF)


# In[251]:



minIDF = len(corpus)*(1/100)  # at least in 1% of the documents
maxIDF = len(corpus)*(90/100)  # maximum in 90% of the documents
minTF = 100 / len(corpus_word) # at least 100 occurencies of the words
df_vocab_useful = df_vocab.loc[(df_vocab.TF > minTF) & (df_vocab.IDF < math.log(len(corpus) / (1 + minIDF)) )& (df_vocab.IDF > math.log(len(corpus) / (1 + maxIDF)) ),]
df_vocab_useful = df_vocab_useful.set_index('Word')

df_vocab_useful["initial_value"] =  0
vocab = df_vocab_useful.loc[: ,("initial_value")].to_dict()


# In[307]:


vocab = df_vocab_useful.loc[: ,("initial_value")].to_dict()


# In[252]:


df_vocab_useful.describe()


# In[324]:


#dic = job.tfidf_voc.iloc[1]
def tf_idf_txt_to_vec(dic, vocab):
    vect = vocab.copy()
    for i, key in enumerate(dic):
        if key in vect.keys():
            vect[key] = dic[key]
    return vect

job["tf_idf_corpus_dic"] = job.tfidf_voc.map(lambda x: tf_idf_txt_to_vec(x,vocab))
job["tf_idf_vec"] = job["tf_idf_corpus_dic"].map(lambda x : list(x.values()))


# In[327]:


job.head()

