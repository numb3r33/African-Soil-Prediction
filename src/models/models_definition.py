from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge

def get_models_by_dataset(dataset_name):
	"""
	Since we will be training models depending on the dataset
	we are associating models with the dataset. So we can pass in the dataset name
	and based on that we will get a list of the models that should be trained on this dataset
	based on our initial experiments
	"""
	
	models = {
		'dataset_1': [
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='rbf'))
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='linear'))
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='poly'))
			]
		],
		
		'dataset_2': [
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='rbf'))
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='linear'))
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='poly'))
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
			],
			[
				('scaler', StandardScaler()),
				('model', (SVR(kernel='poly')))
			]
		],
		'dataset_4': [
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='rbf'))
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='linear'))
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=125, whiten=True)),
				('model', SVR(kernel='poly'))
			]
		],
		'dataset_5': [
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=10, whiten=True)),
				('model', SVR(C=10.0, kernel='rbf'))
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=100, whiten=True)),
				('model', SVR(C=10.0, kernel='linear'))
			],
			[
				('scaler', StandardScaler()),
				('pca', PCA(n_components=100, whiten=True)),
				('model', SVR(C=10.0, kernel='poly'))
			],

		]
	}

	return models[dataset_name]