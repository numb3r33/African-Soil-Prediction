import os, sys

from sklearn.cross_validation import KFold, train_test_split

basepath = os.path.expanduser('~/Desktop/src/African_Soil_Property_Prediction/')
sys.path.append(os.path.join(basepath, 'src'))

from models import eval_metric

def cv_scheme(pipelines, X, y_Ca, y_P, y_Sand, y_SOC, y_pH):
	cv = KFold(len(X), n_folds=5, shuffle=True, random_state=10)
	
	scores = 0
	for itrain, itest in cv:
		Xtr = X.iloc[itrain]
		
		ytr_Ca = y_Ca.iloc[itrain]
		ytr_P = y_P.iloc[itrain]
		ytr_Sand = y_Sand.iloc[itrain]
		ytr_SOC = y_SOC.iloc[itrain]
		ytr_pH = y_pH.iloc[itrain]
		
		Xte = X.iloc[itest]
		
		yte_Ca = y_Ca.iloc[itest]
		yte_P = y_P.iloc[itest]
		yte_Sand = y_Sand.iloc[itest]
		yte_SOC = y_SOC.iloc[itest]
		yte_pH = y_pH.iloc[itest]
	
		pipelines[0].fit(Xtr, ytr_Ca)
		pipelines[1].fit(Xtr, ytr_P)
		pipelines[2].fit(Xtr, ytr_Sand)
		pipelines[3].fit(Xtr, ytr_SOC)
		pipelines[4].fit(Xtr, ytr_pH)
		
		ypred_Ca = pipelines[0].predict(Xte)
		ypred_P = pipelines[1].predict(Xte)
		ypred_Sand = pipelines[2].predict(Xte)
		ypred_SOC = pipelines[3].predict(Xte)
		ypred_pH = pipelines[4].predict(Xte)

		scores += eval_metric.mcrmse([yte_Ca, yte_P, yte_pH, yte_Sand, yte_SOC], [ypred_Ca, ypred_P, ypred_pH, ypred_Sand, ypred_SOC])
	
	return scores / len(cv)


def split_dataset(train_length, **params):
	itrain, itest = train_test_split(range(train_length), **params)

	return itrain, itest
	