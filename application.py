import pandas as pd
from main import ENCODING
from text_processing import tokenization
from learning import get_features_dataset


with open("trainingDataScrapped.json","r", encoding=ENCODING) as file:
    contents = file.read()

ads = pd.read_json(contents)

with open("trained_model.json","r") as file:
    contents = file.read()

trained_model = json.load(contents)

ads["text_process"] = ads['description'].map(tokenization)

vectorization(ads, trained_model['df_vocab_useful'], trained_model['corpus_word_list'])

features = get_features_dataset(ads, trained_model['corpus_word_list'])

for skill in trained_model['model_dict']:
    ads[skill] = trained_model['model_dict'][skill].predict(features)
