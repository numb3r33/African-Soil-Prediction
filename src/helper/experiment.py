import pandas as pd
import numpy as np
import os, sys

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.svm import SVR
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

basepath = os.path.expanduser('~/Desktop/src/African_Soil_Property_Prediction/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(2)

from models import cross_validation, eval_metric, models_definition, find_weights
from helper import utils


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

def save_models(trained_models, dataset_name):
	labels = ['Ca', 'P', 'Sand', 'SOC', 'pH']
	model_names  = ['rbf', 'linear', 'poly']
	
	for i in range(len(labels)):
		for j in range(len(model_names)):
			filepath = 'data/processed/%s/models/%s/%s/%s'%(dataset_name, labels[i], model_names[j], model_names[j])
			joblib.dump(trained_models[i, j], os.path.join(basepath, filepath))

	print('All models written to disk successfully')

if __name__ == '__main__':
	dataset_name = sys.argv[1]

	print('Loading Files\n')
	# load files
	train = pd.read_csv(os.path.join(basepath, 'data/raw/training.csv'))
	test = pd.read_csv(os.path.join(basepath, 'data/raw/sorted_test.csv'))
	
	# load processed dataset
	# let's load a dataset
	train_filepath = 'data/processed/%s/train/train'%dataset_name
	test_filepath  = 'data/processed/%s/test/test'%dataset_name

	train_, test_  = load_dataset(train_filepath, test_filepath)	


	y_Ca, y_P, y_Sand, y_SOC, y_pH = define_target_variables(train)

	# lets get the train and test indices

	params = {
		'test_size' : 0.2,
		'random_state' : 4
	}

	itrain, itest = cross_validation.split_dataset(len(train_), **params)

	X_train, X_test    = utils.get_Xs(train_, itrain, itest) # split the dataset into training and test set.
	y_trains, y_tests  = utils.get_Ys(y_Ca, y_P, y_Sand, y_SOC, y_pH, itrain, itest)

	y_train_Ca, y_train_P, y_train_Sand, y_train_SOC, y_train_pH = y_trains
	y_test_Ca, y_test_P, y_test_Sand, y_test_SOC, y_test_pH      = y_tests

	print('Get models by dataset\n')
	models = models_definition.get_models_by_dataset(dataset_name)

	print('Training Models\n')
	trained_models = train_models(models, [X_train, X_train, X_train, X_train, X_train], \
									  [y_train_Ca, y_train_P, y_train_Sand, y_train_SOC, y_train_pH])

	print('Save Models\n')
	save_models(trained_models, dataset_name)
