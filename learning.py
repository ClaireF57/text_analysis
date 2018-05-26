import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def get_features_dataset(ads, corpus_word_list):
    # Transpose
    features = pd.DataFrame.from_dict(dict(zip(ads.tfidf_voc.index, ads.tfidf_voc.values))).T
    # Name columns
    features.columns = corpus_word_list
    return features


def models_parameters():
    # Parameters to test per model
    parameters_log = [{
        'C': [ 1e02,1e04,1e06,1e08], # learning rate
        'max_iter': [100], # number of epochs
        'solver': ['newton-cg'], # type of solver
        'penalty': ['l2'],
    }]
    parameters_svm = [{
        'alpha': [1e-8,1e-7,1e-6,1e-4], # learning rate
        'epsilon': [1e-4],
        'max_iter': [5,100,1000], # number of epochs
        'loss': ['hinge'], # loss function,
        'penalty': ['l2'],
        'n_jobs': [-1]
    }]
    parameters_NB = [{'alpha': [1e-07,1e-08,1e-09,1e-10], 'fit_prior':['True','False']}]

    return [parameters_svm, parameters_log, parameters_NB]

#  Metrics (accuracy , sensitivity , specificity) about the efficiency of the model
def get_metrics(model,X_test,y_test):
    predicted = model.predict(X_test)
    conf_mx = confusion_matrix(predicted, y_test)
    TN, FP, FN, TP = conf_mx.ravel()

    accuracy = np.sum([TN, TP]) / np.sum([TN, TP, FP, FN])
    sensitivity = TP / np.sum([FN, TP])
    specificity = TN / np.sum([TN, FP])
    metrics = [conf_mx, accuracy, sensitivity, specificity]
    return metrics

# function to automate the research of best estimators for the 3 types of algorithms:
def get_best_model(models, parameters, X_train, y_train, X_test, y_test):

    best_estimators = []
    metrics = []
    for i in range(len(models)):
        ## ! Long function
        gs = GridSearchCV(estimator=models[i], param_grid=parameters[i], scoring='accuracy', cv=10)
        # for models[i] and the parameters to try parameters[i], all type of models are computed
        # to find the model that provide the best scoring "accuracy" calculated with cross validation (cv = 10 folds)
        gs = gs.fit(X_train, y_train)
        # we fit the model with the training dataset
        best_estimators = best_estimators + [gs.best_estimator_]
        # collect the parameters that generated the best model
        metrics = metrics + [get_metrics(gs,X_test, y_test)]
        # collect the metrics for this model on the test set
        results = [best_estimators, metrics]
    return(results)
