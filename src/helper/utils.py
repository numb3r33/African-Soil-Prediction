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
