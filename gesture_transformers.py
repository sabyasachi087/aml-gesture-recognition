from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing.data import StandardScaler


def removeNonNumeric(df):
    return pd.DataFrame(df[df.apply(lambda row : isNumeric(row), axis=1)], dtype=np.float64)

    
def isNumeric(row):
    try:
        row.astype('float')
    except Exception as e:
        print('--------------------NON-NUMERIC-DATA-ERROR--------------------')
        print(row)
        print(e)
        print('-----------------------ERROR-ENDS-HERE------------------------')
        return False
    return True


#--------------------------------End Of Utility Common Functions---------------------------
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Expects a list of panda data frames
        """
        tr = []
        for x in X:
            x = removeNonNumeric(x)
            tr.append(x[self.attribute_names])
        return tr

# -----------------------------End Of DataFrame Selector -------------------------


class CoordinateNormalizer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        tr = []
        for x in X:
#             x = removeNonNumeric(x)
            first_row = x.iloc[0]
            tr.append(x.apply(lambda row: self.norm(row, first_row), axis=1))
        return tr
        
    def norm(self, row, first_row):
        try:
            return row.astype('float') - first_row.astype('float')
        except Exception as e:
            print(row)
            print(e)
            return False
            
# ----------------------- End Of Coordinate Normalizer -------------------


class AccelerometerNormalizer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        tr = []
        for x in X:
#             x = removeNonNumeric(x)
            tr.append(x.apply(lambda row: self.norm(row), axis=1))
        return tr
        
    def norm(self, row):
        return np.sign(row.astype('float'))          

# --------------------------End Of Accelerometer Normalizer---------------------


class AnalogVoltageScaler(BaseEstimator, TransformerMixin):      
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        tr = []
        for x in X:
#             x = removeNonNumeric(x)
            tr.append(self.minMaxNorm(x))
        return tr
    
    def minMaxNorm(self, df):
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result
    
