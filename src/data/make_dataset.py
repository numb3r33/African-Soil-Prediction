import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Data:
    def __init__(self, train, test):
        self.train = train
        self.test = test
    
    def concat_data(self):
        self.data = pd.concat((self.train, self.test), axis=0)
        return self.data
    
    def non_infrared_features(self):
        return self.train.columns[1:-5]
    
    def get_train_test(self):
        mask = self.data.Ca.notnull()
        train = self.data.loc[mask]
        test = self.data.loc[~mask]
        
        return train, test
    
    def encode_categorical_features(self, feature_name):
        lbl = LabelEncoder()
        lbl.fit(self.data[feature_name])
        
        self.data[feature_name] = lbl.transform(self.data[feature_name])
        return self.data[feature_name]
    
    @staticmethod
    def remove_CO2_band(features):
        CO2_band_start = list(features).index('m2379.76')
        CO2_band_end = CO2_band_start + 15
        
        CO2_band = features[CO2_band_start:CO2_band_end]
        
        return features.drop(CO2_band)