
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from collections import Counter
import numpy as np


# In[2]:


job = pd.read_pickle(open("eng_jobs.p","rb"))


# In[3]:


job.shape


# ## Text preprocessing 
# 1- we prepare, clean and tokenize the text 
# 2 - we calculate the tf_idf of each word for each job considering the whole corpus
# 
# ... take some time 

# In[4]:


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


# In[5]:


job["text_process"] = job.text.map(text_preprocessing)
corpus = list(job["text_process"])


# ## text to vec using tf_idf
# 
# - we already vectorized each job ad but they don't share the same vocabulary
# - we have to be able to compare job ads to each other , so we need a global "dictionnary" for all words in the job ads
# - But there are a lot of words so a lot of features 
# - But we are only interested in useful words (neither too common , but frequent enough to bring information that can be used- 
# - to do this we are to study frequency and existence of each words and select a threshold

# In[7]:


#https://stevenloria.com/tf-idf/

import math


def tf(word, txt_tokenized):
    return txt_tokenized.count(word) / len(txt_tokenized)

def n_containing(word, txt_corpus):
    return sum(1 for txt_tokenized in txt_corpus if word in txt_tokenized)

def idf(word, txt_corpus):
    return math.log(len(txt_corpus) / (1 + n_containing(word, txt_corpus)))

#def tfidf(word, txt_tokenized, txt_corpus):
#    return tf(word, txt_tokenized) * idf(word, txt_corpus)




# In[8]:


corpus_word = []
for wordlist in enumerate(corpus):
    corpus_word +=  wordlist[1]
    
corpus_word2 = list(set(corpus_word))


# take 5 minutes

# In[11]:


TF = Counter(corpus_word)
IDF = {word : idf(word, corpus) for word in corpus_word2}


# In[15]:


Term_Freq = pd.DataFrame( {"Word" : list(TF.keys()),"TF" :list(TF.values())} )
Term_IDF = pd.DataFrame( {"Word" : list(IDF.keys()),"IDF" :list(IDF.values())} )
df_vocab = Term_Freq.merge(Term_IDF, on="Word")
df_vocab.TF = df_vocab.TF/sum(df_vocab.TF)


# In[16]:



minIDF = len(corpus)*(1/100)  # at least in 1% of the documents
maxIDF = len(corpus)*(90/100)  # maximum in 90% of the documents
minTF = 100 / len(corpus_word) # at least 100 occurencies of the words
df_vocab_useful = df_vocab.loc[(df_vocab.TF > minTF) & (df_vocab.IDF < math.log(len(corpus) / (1 + minIDF)) )& (df_vocab.IDF > math.log(len(corpus) / (1 + maxIDF)) ),]
df_vocab_useful = df_vocab_useful.set_index('Word')

df_vocab_useful["initial_value"] =  0
vocab = df_vocab_useful.loc[: ,("initial_value")].to_dict()


# In[20]:


df_vocab_useful.head()


# In[54]:


corpus_word_list = list(df_vocab_useful.index)


# In[26]:


df_vocab_useful.loc[word,"IDF"]


# In[ ]:


# get tf.idf per job


# In[132]:


#2min


# In[62]:



job["tfidf_voc"]  = job.text_process.map(lambda txt_tokenized :[tf(word, txt_tokenized)*df_vocab_useful.loc[word,"IDF"] for word in corpus_word_list])




# In[63]:


job.head()


# # skills selection

# In[66]:


job.keyword.unique()


# In[95]:


import heapq
def get_keyword(tfidf_vector_list,nb_keyword):
    C = list(tfidf_vector_list)
    t =[np.mean(x) for x in zip(*C)]
    b =[t.index(number) for number in heapq.nlargest(nb_keyword,t)]
    keywords = list(df_vocab_useful.iloc[b,].index)
    return keywords


# In[104]:



nb_keyword = 5


# In[110]:


tfidf_vector_list = job.loc[job.keyword == "data science", "tfidf_voc"]
k_data = get_keyword(tfidf_vector_list,nb_keyword)
k_data


# In[111]:


tfidf_vector_list = job.loc[job.keyword == "sales", "tfidf_voc"]
k_sales = get_keyword(tfidf_vector_list,nb_keyword)
k_sales


# In[112]:


tfidf_vector_list = job.loc[job.keyword == "supply chain", "tfidf_voc"]
k_sp = get_keyword(tfidf_vector_list,nb_keyword)
k_sp


# In[113]:


tfidf_vector_list = job.loc[job.keyword == "Finance", "tfidf_voc"]
k_fin = get_keyword(tfidf_vector_list,nb_keyword)
k_fin


# In[114]:


#tfidf_vector_list = job.loc[job.keyword == "consulting", "tfidf_voc"]
#k_cons = get_keyword(tfidf_vector_list,nb_keyword)
#k_cons


# In[121]:


skills = k_data + k_sales + k_sp + k_fin 


# In[122]:


len(skills)


# # Training data

# In[116]:


job.head()


# In[123]:


job_training_data = job.copy()


# In[125]:


for skill in skills: 
    job_training_data[skill] = 0
    job_training_data[skill] = job_training_data.text_process.map(lambda text_process : 1 if skill in text_process else 0  )



# In[187]:


job_training_data.shape


# In[188]:


#job_training_data.to_csv("job_training_data.csv",sep="|")
import pickle
pickle.dump(job_training_data, file=open("job_training_data.p", "wb"))


# # Classification SVM

# In[136]:



from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[167]:


# 7 to 27 ...
y = job_training_data.iloc[:,8]
X = pd.DataFrame.from_items(zip(job_training_data.tfidf_voc.index, job_training_data.tfidf_voc.values)).T
X.columns =list(df_vocab_useful.index)


# In[168]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[169]:


print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[170]:


from sklearn.linear_model import SGDClassifier
#https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)


# In[171]:


svm


# In[172]:


svm_model =  svm.fit(X_train,y_train)


# In[173]:


predicted_svm = svm_model.predict(X_test)


# In[174]:


np.mean(predicted_svm == y_test)


# In[176]:


y = job_training_data.iloc[:,8]
def pred_accu_skill (y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    svm_model =  svm.fit(X_train,y_train)
    predicted_svm = svm_model.predict(X_test)
    accuracy = np.mean(predicted_svm == y_test)
    return accuracy


# In[185]:


for i in range(7, 27):
    y = job_training_data.iloc[:,i]
    print(pred_accu_skill(y))

