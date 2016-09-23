import pandas as pd
import numpy as np
import os, sys

import warnings
warnings.filterwarnings('ignore')

basepath = os.path.expanduser('~/Desktop/src/African_Soil_Property_Prediction/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(2)

from models import cross_validation, eval_metric, models_definition, find_weights
from helper import utils

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

	train_, test_  = utils.load_dataset(train_filepath, test_filepath)	


	y_Ca, y_P, y_Sand, y_SOC, y_pH = utils.define_target_variables(train)

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
	trained_models = utils.train_models(models, [X_train, X_train, X_train, X_train, X_train], \
									  [y_train_Ca, y_train_P, y_train_Sand, y_train_SOC, y_train_pH])

	print('Save Models\n')
	utils.save_models(trained_models, dataset_name)
