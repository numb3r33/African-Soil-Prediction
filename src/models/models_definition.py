from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

def get_models_by_dataset(dataset_name):
	"""
	Since we will be training models depending on the dataset
	we are associating models with the dataset. So we can pass in the dataset name
	and based on that we will get a list of the models that should be trained on this dataset
	based on our initial experiments

	Dataset 1: Contains all the infrared measurements
	Dataset 2: Contains all features other than CO2 band
	Dataset 3: Contains the spatial features
	Dataset 4: Grouped values by the wavelength
	Dataset 5: Features chosen by feature relevance 
	Dataset 6: T-sne on the wavebands and remove features with low standard deviation
	"""
	
	models = {
		'dataset_1': [
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', Ridge())
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='linear'))
			]
		],
		
		'dataset_2': [
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', Ridge())
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='linear'))
			]
		],
		'dataset_3': [
			[
				('scaler', StandardScaler()),
				('model', (SVR(kernel='rbf')))
			],
			[
				('scaler', StandardScaler()),
				('model', (SVR(kernel='linear')))
			]
		],
		'dataset_4': [
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', Ridge())
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='linear'))
			]
		],
		'dataset_5': [
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', Ridge())
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(C=1.0, kernel='linear'))
			]
		],
		'dataset_6': [
			[
				('scaler', StandardScaler()),
				('model', Ridge())
			],
			[
				('model', RandomForestRegressor(n_estimators=250, max_depth=15, n_jobs=-1))
			],
		]
	}

	return models[dataset_name]