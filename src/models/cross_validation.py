import os, sys

from sklearn.cross_validation import KFold, train_test_split
from sklearn.pipeline import Pipeline

basepath = os.path.expanduser('~/Desktop/src/African_Soil_Property_Prediction/')
sys.path.append(os.path.join(basepath, 'src'))

from models import eval_metric

def cv_scheme(model, Xs, ys):
	"""
	model : Model definition
	Xs    : List of the dataframes for each of the target variables
	ys    : List of the target variables

	"""

	# TODO: hard code this configuration as of now, will find a way to modify this.
	cv = KFold(len(Xs[0]), n_folds=10, shuffle=True, random_state=10)
	
	# label names 
	label_names = ['Ca', 'P', 'Sand', 'SOC', 'pH']
	
	model = Pipeline(model)
		
	scores = 0
	index = 0

	for itrain, itest in cv:
		y_true = []
		y_pred = []
		print('Fold: %d'%index)
		index  += 1

		for i in range(len(ys)):
			Xtr = Xs[i].iloc[itrain]
			ytr = ys[i].iloc[itrain]

			Xte = Xs[i].iloc[itest]
			yte = ys[i].iloc[itest]

			model.fit(Xtr, ytr)

			y_true.append(yte)
			pred = model.predict(Xte)

			print('MCRMSE score for label: %s -> %f'%(label_names[i], eval_metric.mcrmse([yte], [pred])))
			y_pred.append(model.predict(Xte))

		score = eval_metric.mcrmse(y_true, y_pred)
		print('\nMCRMSE score: %f\n'%score)
		scores += score

	return scores / len(cv)




def split_dataset(train_length, **params):
	itrain, itest = train_test_split(range(train_length), **params)

	return itrain, itest
	