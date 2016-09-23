import numpy as np
import os, sys

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.svm import SVR
from sklearn.externals import joblib

basepath = os.path.expanduser('~/Desktop/src/African_Soil_Property_Prediction/')
sys.path.append(os.path.join(basepath, 'src'))



def remove_CO2_band(features):
	"""
	Takes in a list of features,
	looks for the start and end of the C02 band
	and remove that band from the featureset.
	"""

	CO2_band_start = list(features).index('m2379.76')
	CO2_band_end = CO2_band_start + 15
	
	CO2_band = features[CO2_band_start:CO2_band_end]
	
	return features.drop(CO2_band)

def get_Xs(X, itrain, itest):
	X_train = X.iloc[itrain]
	X_test  = X.iloc[itest]
	
	return X_train, X_test
	
def get_Ys(y_Ca, y_P, y_Sand, y_SOC, y_pH, itrain, itest):
	y_train_Ca = y_Ca.iloc[itrain]
	y_test_Ca  = y_Ca.iloc[itest]
	
	y_train_P  = y_P.iloc[itrain]
	y_test_P  = y_P.iloc[itest]
	
	y_train_Sand  = y_Sand.iloc[itrain]
	y_test_Sand  = y_Sand.iloc[itest]
	
	y_train_SOC  = y_SOC.iloc[itrain]
	y_test_SOC  = y_SOC.iloc[itest]
	
	y_train_pH  = y_pH.iloc[itrain]
	y_test_pH  = y_pH.iloc[itest]
	
	
	return ([y_train_Ca, y_train_P, y_train_Sand, y_train_SOC, y_train_pH],
			[y_test_Ca, y_test_P, y_test_Sand, y_test_SOC, y_test_pH])

def predict_targets(trained_models, Xs):
    """
    trained_models : List of the trained models
    Xs             : Held out examples for each of the target variables
    """
    
    n_target = len(Xs)
    n_models = len(trained_models[0])
    
    predictions = np.empty((n_target, n_models), dtype=np.ndarray)
    
    for i in range(n_target):
        for j in range(n_models):
            predictions[i, j] = trained_models[i, j].predict(Xs[i])
        
    return predictions


# load a dataset
def load_dataset(train_filepath, test_filepath):
	train_    = joblib.load(os.path.join(basepath, train_filepath))
	test_     = joblib.load(os.path.join(basepath, test_filepath))
	
	return train_, test_

# define target variables
def define_target_variables(train):    
	
	y_Ca    = train.Ca
	y_P     = train.P
	y_Sand  = train.Sand
	y_SOC   = train.SOC
	y_pH    = train.pH
	
	return y_Ca, y_P, y_Sand, y_SOC, y_pH

# train models for all of the target varibles
def train_models(models, Xs, ys):
	"""
	models : List of models that should be trained on a given (X, y)
	Xs     : List of feature set for all of the target variables
	ys     : List of the target variables
	"""
	
	n_target = len(ys)
	n_models = len(models)
	
	trained_models = np.empty((n_target, n_models), dtype=Pipeline)
	
	for i in range(n_target):
		for j in range(n_models):
			trained_models[i, j] = models[j].fit(Xs[i], ys[i])

	return trained_models

# save the trained models against a dataset name
def save_models(trained_models, dataset_name):
	labels = ['Ca', 'P', 'Sand', 'SOC', 'pH']
	model_names  = ['rbf', 'linear', 'poly']
	
	for i in range(len(labels)):
		for j in range(len(model_names)):
			filepath = 'data/processed/%s/models/%s/%s/%s'%(dataset_name, labels[i], model_names[j], model_names[j])
			joblib.dump(trained_models[i, j], os.path.join(basepath, filepath))

	print('All models written to disk successfully')


# load the models trained on a dataset for every target variable
def load_models(dataset_name):
	labels      = ['Ca', 'P', 'Sand', 'SOC', 'pH']
	model_names = ['rbf', 'linear', 'poly']

	n_target = len(labels)
	n_models = len(model_names)

	trained_models = np.empty((n_target, n_models), dtype=Pipeline)

	for i in range(n_target):
		for j in range(n_models):
			filepath = 'data/processed/%s/models/%s/%s/%s'%(dataset_name, labels[i], model_names[j], model_names[j])
			trained_models[i, j] = joblib.load(os.path.join(basepath, filepath))

	return trained_models