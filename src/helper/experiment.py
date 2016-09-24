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
	split_dataset = sys.argv[2]

	print('Loading Files\n')
	# load files
	train = pd.read_csv(os.path.join(basepath, 'data/raw/training.csv'))
	test = pd.read_csv(os.path.join(basepath, 'data/raw/sorted_test.csv'))
	
	
	# labels
	labels = ['Ca', 'P', 'Sand', 'SOC', 'pH'] # take from command line

	trains_, tests_  = utils.load_datasets(dataset_name, labels)	

	y_Ca, y_P, y_Sand, y_SOC, y_pH = utils.define_target_variables(train)

	# lets get the train and test indices

	if split_dataset == 'yes':
		params = {
			'test_size' : 0.2,
			'random_state' : 4
		}

		itrain, itest = cross_validation.split_dataset(len(train), **params)

		X_train_Ca, X_test_Ca        = utils.get_Xs(trains_[0], itrain, itest) 
		X_train_P, X_test_P          = utils.get_Xs(trains_[1], itrain, itest) 
		X_train_Sand, X_test_Sand    = utils.get_Xs(trains_[2], itrain, itest) 
		X_train_SOC, X_test_SOC      = utils.get_Xs(trains_[3], itrain, itest) 
		X_train_pH, X_test_pH        = utils.get_Xs(trains_[4], itrain, itest)

		X_trains = [X_train_Ca, X_train_P, X_train_Sand, X_train_SOC, X_train_pH]
		X_tests  = [X_test_Ca, X_test_P, X_test_Sand, X_test_SOC, X_test_pH]

		y_trains, y_tests  = utils.get_Ys(y_Ca, y_P, y_Sand, y_SOC, y_pH, itrain, itest)

		y_train_Ca, y_train_P, y_train_Sand, y_train_SOC, y_train_pH = y_trains
		y_test_Ca, y_test_P, y_test_Sand, y_test_SOC, y_test_pH = y_tests
	
	else:
		X_trains = trains_
		X_tests  = tests_ 

		y_trains = [y_Ca, y_P, y_Sand, y_SOC, y_pH]

	print('Get models by dataset\n')
	models = models_definition.get_models_by_dataset(dataset_name)

	print('Training Models\n')
	
	model_names = ['rbf', 'linear', 'poly'] # take from command line
	test_preds = np.empty((len(labels), len(model_names)), dtype=np.ndarray)

	for i in range(len(labels)):
		for j in range(len(model_names)):
			model = utils.train_model(models[j], X_trains[i], y_trains[i], dataset_name, labels[i], model_names[j])
			test_preds[i, j] = utils.predict_targets(model, X_tests[i])
		