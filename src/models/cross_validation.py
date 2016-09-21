import os, sys

from sklearn.cross_validation import KFold, train_test_split

basepath = os.path.expanduser('~/Desktop/src/African_Soil_Property_Prediction/')
sys.path.append(os.path.join(basepath, 'src'))

from models import eval_metric

def cv_scheme(pipelines, Xs, ys):
	cv = KFold(len(Xs[0]), n_folds=10, shuffle=True, random_state=10)
	
	scores = 0
	index = 0

	for itrain, itest in cv:
		print('======== Iteration: %d\n'%index)
		index   = index + 1

		y_preds = []
		y_true  = []
		
		for i in range(len(pipelines)):
			Xtr = Xs[i].iloc[itrain]
			ytr = ys[i].iloc[itrain]

			Xte = Xs[i].iloc[itest]
			yte = ys[i].iloc[itest]

			pipelines[i].fit(Xtr, ytr)
			pred = pipelines[i].predict(Xte)
			
			y_true.append(yte)
			y_preds.append(pred)

		score = eval_metric.mcrmse(y_true, y_preds)
		print('MCRMSE score: %f\n'%score)

		scores = scores + score
	
	return scores / len(cv)


def split_dataset(train_length, **params):
	itrain, itest = train_test_split(range(train_length), **params)

	return itrain, itest
	