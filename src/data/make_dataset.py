import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Data:
    def __init__(self, train, test, remove_CO2_features=False):
        self.train = train
        self.test = test
        self.remove_CO2_features = remove_CO2_features
    
    def concat_data(self):
        self.data = pd.concat((self.train, self.test), axis=0)
        return self.data

    def prepare(self):
        features = self.non_infrared_features()
        self.concat_data()
        self.encode_categorical_features('Depth')
        train_, test_ = self.get_train_test()
        
        if self.remove_CO2_features:
            features = self.remove_CO2_band(features)
            train_ = train_[features]
            test_ = test_[features]

        return train_[features], test_[features]

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