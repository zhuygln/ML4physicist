import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
###### Read in data
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
##################################################
import pickle
from DateTime import DateTime
import time

current_time = DateTime(time.time(), 'US/Eastern')
framework = 'autosklearn'
datasetn = 'bankmarketing'
foldn =  '3'
dirt = '../'

resultfile = str(framework) + str(datasetn) + str(foldn) + \
str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
str(current_time.h_24()) + str(current_time.minute())  + str(time.time())+'.txt'


data =pd.read_csv("/home/test/bank.csv",delimiter=';')
column= data.columns.values
X = data[column[:-1]]
y = data[column[-1]]
lb = preprocessing.LabelBinarizer()
y= lb.fit_transform(y)
##################################################
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),\
    ('scaler', StandardScaler())])
numeric_features = ['age','balance','day','duration','pdays','previous']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'campaign', 'poutcome']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
     ('cat', categorical_transformer, categorical_features)])
################################################################################
X_train, X_test, y_train, y_test = \
sklearn.model_selection.train_test_split(X, y, random_state=1)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.fit_transform(X_test)
print(X_test)
#######################################################################################
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=360,\
        delete_tmp_folder_after_terminate=False,\
        resampling_strategy='cv',\
        resampling_strategy_arguments={'folds': int(foldn)},)

##########################################

automl.fit(X_train, y_train)
automl.refit(X_train, y_train)
y_pred = automl.predict(X_test)
print("accuracy: ", sklearn.metrics.accuracy_score(automl.predict(X_train), y_train))
print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
print(automl.sprint_statistics())
print(automl.show_models())
#pattern = r"(?P<framework>[\w\-]+?)_(?P<task>[\w\-]+)_(?P<fold>\d+)(_(?P<datetime>\d{8}T\d{6}))?.csv"

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
# sample usage
save_object(automl.show_models(),str("showmodels")+resultfile)

finalmodel_file ='finalmodelemsenble.pkl'

from sklearn.externals import joblib
_ = joblib.dump(automl,finalmodel_file, compress=9)

import coremltools
import sklearn
sklearn_model = sklearn.externals.joblib.load(finalmodel_file)
coreml_model = coremltools.converters.sklearn.convert(sklearn_model)
coreml_model.save(finalmodel_file + '.mlmodel')

#print("automodel1",automl.show_models())
#print(type(automl.show_models()))
#print(automl.cv_results_)
print("automl.fit")
print(automl.fit(X_test, y_test))
kfold = KFold(n_splits=3, random_state=0)
results = cross_val_score(automl, X_train, y_train, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#finalmodel = open(finalmodel_file,'wb')
#pickle.dump((automl,X_train,y_train,results),finalmodel)
#finalmodel.close()
