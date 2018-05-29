import numpy as np
import pandas as pd
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from learning import get_features_dataset, models_parameters, get_best_model, get_metrics
from text_processing import define_corpus, tokenization, vectorization, get_keywords, get_skills_list, specify_skill

# Parameters
ENCODING = 'ISO 8859-1'
NB_KEYWORDS = 5
OKGREEN = '\033[92m'
ENDC = '\033[0m'

# Opens and read the json data file
with open("trainingDataScrapped.json","r", encoding=ENCODING) as file:
    contents = file.read()

ads = pd.read_json(contents)

## For test cases, take only a part of the set -- TODO Remove in production
small_set = pd.DataFrame()
for searchTerm in ads.searchTerm.unique():
    small_set = small_set.append(ads.loc[ads.searchTerm == searchTerm].head(100))
ads = small_set

##
## Define corpus
print(OKGREEN, "[Define corpus]", ENDC)
df_vocab_useful, corpus_word_list = define_corpus(ads)
vectorization(ads, df_vocab_useful, corpus_word_list)
## end define corpus

# Get training data set
print(OKGREEN, "[Get training dataset]", ENDC)
skills_list = get_skills_list(ads, NB_KEYWORDS, df_vocab_useful)
FIRST_SKILL_COLUMN_NB = len(ads.columns)
LAST_SKILL_COLUMN_NB = FIRST_SKILL_COLUMN_NB + len(skills_list)

specify_skill(ads, skills_list)
# end get training data set
## SAVE training ads
print(OKGREEN, "[Get features dataset]", ENDC)
features = get_features_dataset(ads, corpus_word_list)

# Initialization of models with default parameters
svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
log = LogisticRegression(penalty='l2' ,random_state= 42, max_iter=100)
NB = MultinomialNB()

PARAMETERS = models_parameters()

## get the best model for each skills to classify
print(OKGREEN, "[Train]", ENDC)
model_dict = dict()
for i in range(FIRST_SKILL_COLUMN_NB, LAST_SKILL_COLUMN_NB):
    y = ads.iloc[:,i]
    # We generate new train/test set (because Y is changing)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3)

    # Initialization of the models
    svm_model =  svm.fit(X_train,y_train)
    log_model = log.fit(X_train,y_train)
    NB_model = NB.fit(X_train,y_train)
    models = [svm_model, log_model, NB_model]

    # find the best model for the specific skill "i"
    results = get_best_model(models, PARAMETERS, X_train, y_train, X_test,y_test)

    # select the type of model which accuracy is the highest
    best_model = results[0][np.argmax([results[1][i][2] for i in range(len(results[1]))])-1]

    # save the best model for the specific skill in a dictionnary
    model_dict[ads.columns.values[i]] = best_model

    # check the prediction skills of the best model
    print(get_metrics(best_model, X_test, y_test))

# Dump the results in a file
with open('trained_model.p', 'wb') as outfile:
    pickle.dump({'model_dict': model_dict,
               'corpus_word_list': corpus_word_list,
               'df_vocab_useful': df_vocab_useful
               }, outfile)
