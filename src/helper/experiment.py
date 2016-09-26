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
	"""
	Responsibility of this module is to train models on given dataset and
	store them on disk.

	Different modes in which this routine can be run is 
	
	1. Train on full dataset.
	2. Do cross validation to see how the models are performing on the dataset.

	Store the trained models on the disk, return the predictions.

	"""

	dataset_name     = sys.argv[1]
	cv               = sys.argv[2]

	# load datasets
	train = pd.read_csv(os.path.join(basepath, 'data/raw/training.csv'))
	test  = pd.read_csv(os.path.join(basepath, 'data/raw/sorted_test.csv'))


	# label names
	labels      = ['Ca', 'P', 'Sand', 'SOC', 'pH']
	models      = models_definition.get_models_by_dataset(dataset_name)
	model_names = ['ridge', 'linear']
	n_models    = len(models)

	X_trains, X_tests = utils.load_datasets(dataset_name, labels)

	y_Ca, y_P, y_Sand, y_SOC, y_pH = utils.define_target_variables( train )

	# check if we need to do cross validate
	if cv == 'yes':
		
		params = {
			'test_size': 0.2,
			'random_state': 3
		}

		itrain, itest = cross_validation.split_dataset(len(train), **params)


		X_train_Ca, X_test_Ca      = utils.get_Xs(X_trains[0], itrain, itest)
		X_train_P, X_test_P        = utils.get_Xs(X_trains[1], itrain, itest)
		X_train_Sand, X_test_Sand  = utils.get_Xs(X_trains[2], itrain, itest)
		X_train_SOC, X_test_SOC    = utils.get_Xs(X_trains[3], itrain, itest)
		X_train_pH, X_test_pH      = utils.get_Xs(X_trains[4], itrain, itest)

		X_trains = [X_train_Ca, X_train_P, X_train_Sand, X_train_SOC, X_train_pH]
		X_tests  = [X_test_Ca, X_test_P, X_test_Sand, X_test_SOC, X_test_pH]

		y_trains, y_tests = utils.get_Ys(y_Ca, y_P, y_Sand, y_SOC, y_pH, itrain, itest)

		for i in range(len(models)):
			print('Model: %s\n'%model_names[i])
			cv_score = cross_validation.cv_scheme(models[i], X_trains, y_trains)
			print('Mean cross validations score: %f\n'%cv_score)
		
	else:
		y_trains = [y_Ca, y_P, y_Sand, y_SOC, y_pH]

	for i in range(len(labels)):
		for j in range(n_models):
			model         = models[j]
			trained_model = utils.train_and_save(model, X_trains[i], y_trains[i], dataset_name, labels[i], model_names[j])
			
			if cv == 'yes': # only test when we are in this mode else just save
				mcrmse = eval_metric.mcrmse([y_tests[i]], [trained_model.predict(X_tests[i])])
				print('MCRMSE on the unseen examples for label: %s for model: %s is %f'%(labels[i], model_names[j], mcrmse))
