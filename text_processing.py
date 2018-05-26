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


# ## Text preprocessing
# 1- we prepare, clean and tokenize the text
# 2 - we calculate the tf_idf of each word for each job considering the whole corpus
#
# ... take some time

#https://www.quora.com/How-can-I-tokenize-a-string-without-using-built-in-function

def tokenization(text):
    WORD = re.compile(r'\w+')
    nlp = WORD.findall(text)

    # remove non alphanumeric character inculding punctuation
    nlp = [re.sub('[^A-Za-z]', '', word) for word in nlp]

    # remove stop words ; get each word in lowercase ; stem each word
    stop_words = set(stopwords.words('english'))
    stop_words = list(stop_words) + [''] # add other words to remove
    porter_stemmer = PorterStemmer()
    preprocessed = [porter_stemmer.stem(word.lower()) for word in nlp]
    preprocessed = [word for word in preprocessed if not word in stop_words]
    return preprocessed

    # preprocessed_Series = pd.Series(preprocessed)
    # if df_vocab_useful:
    #     preprocessed_Series = preprocessed_Series.map(lambda txt_tokenized :[tf(word, txt_tokenized) * df_vocab_useful.loc[word, "IDF"] for word in corpus_word_list])

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

def define_corpus(ads):
    ads["text_process"] = ads['description'].map(tokenization)
    corpus = list(ads["text_process"])

    # Take only unique words
    corpus_word = []
    for wordlist in enumerate(corpus):
        corpus_word +=  wordlist[1]

    corpus_word2 = list(set(corpus_word))

    # TF IDF
    TF = Counter(corpus_word)
    # Why does IDF give the same negative number to all words?
    IDF = {word : idf(word, corpus) for word in corpus_word2}

    # Transform previous TF and IDF in DataFrames
    Term_Freq = pd.DataFrame( {"Word" : list(TF.keys()), "TF" :list(TF.values())} )
    Term_IDF = pd.DataFrame( {"Word" : list(IDF.keys()), "IDF" :list(IDF.values())} )

    # Merge both based on word
    df_vocab = Term_Freq.merge(Term_IDF, on="Word")
    df_vocab.TF = df_vocab.TF / sum(df_vocab.TF)

    # Set filters
    minIDF = len(corpus) * (1 / 100)  # at least in 1% of the documents
    maxIDF = len(corpus) * (90 / 100)  # maximum in 90% of the documents
    minTF = 100 / len(corpus_word) # at least 100 occurences of the words

    # Implement the filters
    df_vocab_useful = df_vocab.loc[(df_vocab.TF > minTF) & (df_vocab.IDF < math.log(len(corpus) / (1 + minIDF)) )& (df_vocab.IDF > math.log(len(corpus) / (1 + maxIDF)) ),]
    df_vocab_useful = df_vocab_useful.set_index('Word')

    corpus_word_list = list(df_vocab_useful.index)

    return df_vocab_useful, corpus_word_list
    ## END DEFINE CORPUS

def vectorization(ads, df_vocab_useful, corpus_word_list):
    ads["tfidf_voc"]  = ads.text_process.map(lambda txt_tokenized :[tf(word, txt_tokenized) * df_vocab_useful.loc[word, "IDF"] for word in corpus_word_list])

def get_skills_list(ads, nb_keywords, df_vocab_useful):
    domain_skills = []
    for searchTerm in ads.searchTerm.unique():
        # Use 100 ads in domain searchTerm
        tfidf_vector_list = ads.loc[ads.searchTerm == searchTerm, "tfidf_voc"].iloc[:100,]
        domain_skills += get_keywords(tfidf_vector_list, nb_keywords, df_vocab_useful)

    return set(domain_skills)

def specify_skill(ads, skills_list):
    for skill in skills_list:
        ads[skill] = ads.text_process.map(lambda text_process : 1 if skill in text_process else 0)
