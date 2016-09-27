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

def predict_target(trained_model, X):
	"""
	trained_model  : Trained Model.
	X             : Held out examples.
	"""
	return trained_model.predict(X)
	

# load a dataset
def load_dataset(train_filepath, test_filepath):
	train_    = joblib.load(os.path.join(basepath, train_filepath))
	test_     = joblib.load(os.path.join(basepath, test_filepath))
	
	return train_, test_

def load_datasets(dataset_name, labels):
	trains_ = []
	tests_  = []

	for i in range(len(labels)):
	    # let's load a dataset
	    train_filepath = 'data/processed/%s/%s/train/train'%(dataset_name, labels[i])
	    test_filepath  = 'data/processed/%s/%s/test/test'%(dataset_name, labels[i])

	    train_, test_  = load_dataset(train_filepath, test_filepath)
	    
	    trains_.append(train_)
	    tests_.append(test_)

	return trains_, tests_

# define target variables
def define_target_variables(train):    
	
	y_Ca    = train.Ca
	y_P     = train.P
	y_Sand  = train.Sand
	y_SOC   = train.SOC
	y_pH    = train.pH
	
	return y_Ca, y_P, y_Sand, y_SOC, y_pH

# train models for all of the target varibles
def train_and_save(model, Xs, ys, dataset_name, label_name, model_name):
	"""
	models : List of models that should be trained on a given (X, y)
	Xs     : List of feature set for all of the target variables
	ys     : List of the target variables
	"""
	
	model = Pipeline(model)
	model = model.fit(Xs, ys)
	save_model(model, dataset_name, label_name, model_name)

	return model # return back for prediction

# save the trained models against a dataset name
def save_model(trained_model, dataset_name, label_name, model_name):
	filepath = 'data/processed/%s/%s/models/%s/%s'%(dataset_name, label_name, model_name, model_name)
	joblib.dump(trained_model, os.path.join(basepath, filepath))

	print('Model saved successfully')


def get_labels():
	return ['Ca', 'P', 'Sand', 'SOC', 'pH']