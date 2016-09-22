import numpy as np
from scipy.optimize import nnls


def weight_selected(data, labels):
    weights, _ = nnls(data[:len(labels)], labels)
    return weights


def stack_predictions(preds):
	# preds contain list of predictions for a given target variable
	return np.vstack(preds).T
	
def calculate_weights(y_true, y_pred):
	return weight_selected(y_pred, y_true)

def balance_predictions(y_test, y_pred, weights):
	return y_pred[:, weights > 0].mean(axis=1)[:len(y_test)]


def find(y_test, y_pred):
	"""
	y_test: list of predictions for a given target variable
	y_pred: list of predictions spit out by the model
	
	Returns: weights and the balanced prediction
	"""

	stacked_predictions = stack_predictions(y_pred)
	weights = calculate_weights(y_test, stacked_predictions)

	return weights, balance_predictions(y_test, stacked_predictions, weights)

