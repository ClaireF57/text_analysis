
## Classification SVM

#librairies requirements
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# Source of inspiration
#https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a



# get the already prepared data
# columns : array(['text', '_id', 'keyword', 'title', 'lang', 'text_process','tfidf_voc',
#       'data', 'scienc', 'research', 'technic', 'model',
#       'sale', 'account', 'hotel', 'custom', 'client', 'airbu', 'suppli',
#       'chain', 'logist', 'faurecia', 'financ', 'financi', 'bank',
#       'control', 'natixi'], dtype=object)

filename = "job_training_data.p"
job_training_data = pickle.load(open(filename, 'rb'))


# Test and optimization of classifiers

# Training / test set

# features dataset
X = pd.DataFrame.from_items(zip(job_training_data.tfidf_voc.index, job_training_data.tfidf_voc.values)).T
# X.columns =list(df_vocab_useful.index) # not necessary but could be cool to understand better ...

# Initialization of models with default parameters
svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
log = LogisticRegression(penalty='l2' ,random_state= 42, max_iter=100)
NB = MultinomialNB()

## Let's optimize the parameter

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

parameters = [ parameters_svm, parameters_log, parameters_NB]


# function to get metrics (accuracy , sensitivity , speicificity) about the efficiency of the model


def get_metrics(model,X_test,y_test):
    predicted = model.predict(X_test)
    conf_mx = confusion_matrix(predicted, y_test)
    TN, FP, FN, TP = conf_mx.ravel()

    accuracy = (TN + TP) / (TN + TP + FP + FN)
    sensitivity = TP / (FN + TP)
    specificity = TN / (TN + FP)
    metrics = [conf_mx, accuracy, sensitivity, specificity]
    return metrics


# function to automate the research of best estimators for the 3 types of algorithms:


def get_best_model(models, parameters, X_train, y_train, X_test,y_test):

    best_estimators = []
    metrics = []
    for i in range(len(models)):
        gs = GridSearchCV(estimator = models[i], param_grid = parameters[i], scoring = 'accuracy',cv =10)
        #for models[i] and the parameters to try parameters[i], all type of models are computed
        # to find the model that provide the best scoring "accuracy" calculated of cross validation (cv = 10 folds)
        gs = gs.fit(X_train, y_train)
        # we fit the model with the training dataset
        best_estimators = best_estimators + [gs.best_estimator_]
        # collect the parameters that generated the best model
        metrics = metrics + [get_metrics(gs,X_test, y_test)]
        # collect the metrics for this model on the test set
        results = [best_estimators, metrics]
    return(results)


## get the best model for each skills to classify

model_dict = dict()
for i in range(7, 27):
    y = job_training_data.iloc[:,i] # for each skill , going from column 7 to 27
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # we generate new train/test set (because Y is changing)

    # first Initialization of the models
    svm_model =  svm.fit(X_train,y_train)
    log_model = log.fit(X_train,y_train)
    NB_model = NB.fit(X_train,y_train)
    models = [svm_model, log_model, NB_model]

    # find the best model for the specific skill "i"
    results = get_best_model(models, parameters, X_train, y_train, X_test,y_test)

    # select the type of model which accuracy is the highest
    best_model = results[0][np.argmax([results[1][i][2] for i in range(len(results[1]))])-1]

    # save the best model for the specific skill in a dictionnary
    model_dict[job_training_data.columns.values[i]] = best_model

    # check the prediction skills of the best model
    print(get_metrics(best_model,X_test,y_test))


## save the models
# save the dict
filename = "model_dict.p"
pickle.dump(model_dict, open(filename, 'wb'))


# upload the models 
## to try the test

filename = "model_dict.p"
model_dico = pickle.load(open(filename, 'rb'))
y = job_training_data.iloc[:,7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("let's test the model to classify if data is in the jobs ads or not")
print(model_dico['data'].get_metrics(X_test, y_test))
