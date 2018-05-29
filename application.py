import pandas as pd
import pickle
import json
import random
from text_processing import tokenization, vectorization
from learning import get_features_dataset

ENCODING = 'ISO 8859-1'

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

with open("trainingDataScrapped.json","r", encoding=ENCODING) as file:
    contents = file.read()

ads_set = pd.read_json(contents)

result = []
random.seed()
bounds = random.randrange(len(ads_set))
for i in range(bounds, bounds + 20):
    ads = ads_set[i:i + 1]
    trained_model = pickle.load(open("trained_model.p", 'rb'))

    ads["text_process"] = ads['description'].map(tokenization)

    vectorization(ads, trained_model['df_vocab_useful'], trained_model['corpus_word_list'])

    features = get_features_dataset(ads, trained_model['corpus_word_list'])

    for skill in trained_model['model_dict']:
        ads[skill] = trained_model['model_dict'][skill].predict(features)
    ads_t = ads.T
    ads_t.columns = ['value']
    vector = ads_t.loc[ads_t['value'] == 1]
    skills = vector.T.columns.values
    # description = ads.iloc[0]
    # print(description['description'])
    intersection = list(set(skills) & set(ads.iloc[0]['text_process']))

    if not len(intersection) == len(skills):
        result.append({'skills_proposed': diff(skills, ads.iloc[0]['text_process']),
                       'description': ads.iloc[0]['description']})

with open('./results.json', 'w') as outfile:
   json.dump(result, outfile)
