from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor

def get_models_by_dataset(dataset_name):
    """
    Since we will be training models depending on the dataset
    we are associating models with the dataset. So we can pass in the dataset name
    and based on that we will get a list of the models that should be trained on this dataset
    based on our initial experiments
    """
    
    models = {
        'dataset_1': [
            Pipeline([
                ('pca', PCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='rbf'))
            ]),
            Pipeline([
                ('pca', PCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='linear'))
            ]),
            Pipeline([
                ('pca', PCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='poly'))
            ])
        ],
        
        'dataset_2': [
            Pipeline([
                ('pca', PCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='rbf'))
            ]),
            Pipeline([
                ('pca', PCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='linear'))
            ]),
            Pipeline([
                ('pca', PCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='poly'))
            ])
        ],
        'dataset_3': [
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', (SVR(kernel='rbf')))
            ]),
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', (SVR(kernel='linear')))
            ]),
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', (SVR(kernel='poly')))
            ])
        ],
        'dataset_4': [
            Pipeline([
                ('pca', RandomizedPCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='rbf'))
            ]),
            Pipeline([
                ('pca', RandomizedPCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='linear'))
            ]),
            Pipeline([
                ('pca', RandomizedPCA(n_components=125, whiten=True)),
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='poly'))
            ])
        ]
    }

    return models[dataset_name]