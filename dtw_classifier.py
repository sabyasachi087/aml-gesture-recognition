from sklearn.base import BaseEstimator, ClassifierMixin
from dtw import fastdtw
import numpy as np


class DTWClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, neighbours=3, dist='minkowski'):
        """
        dist: Distance metrics - euclidean, minkowski
        neighbours: Nearest neighbours
        """       
        self.neighbours = neighbours
        self.dist = dist

    def fit(self, X, y):
        self.X = X
        self.y = np.array(y)
        return self

    def predict(self, x_test):
        pred = []; err = []
        for id_test in range(len(x_test)):
            result = np.zeros(len(self.y))
            for id_train in range(len(self.X)):
                try:
                    min_dist = fastdtw(self.X[id_train], x_test[id_test], self.dist)
                    result[id_train] = min_dist
                except Exception as e:
                    print(self.y[id_train]) 
                    print(self.X[id_train]) 
                    print(x_test[id_test])    
                    print(e)       
            res_indx = result.argsort()[:self.neighbours]
            pred.append(self.y[res_indx])
            err.append(result[res_indx])
        return pred, err
        
    def score(self, X, y=None):
        y_predict = self.predict(X)
        count = 0
        for idx in range(len(y)):
            if y[idx] == y_predict[idx][0]:
                count += 1
        return count / len(y)
