from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

# -----------------------------End Of DataFrame Selector -------------------------


class CoordinateNormalizer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        first_row = X.iloc[0]
        return X.apply(lambda row: self.norm(row, first_row), axis=1)
        
    def norm(self,row,first_row):
        try:
            return row.astype('float') - first_row.astype('float')
        except:
            print(row)
            return row
            
        
        
        
        
        
        
        
        
        
