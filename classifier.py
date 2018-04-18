import numpy as np
import pandas as pd
import curator as crt
from DataType import DataType
from dtw import fastdtw
from gesture_transformers import CoordinateNormalizer, AccelerometerNormalizer, DataFrameSelector, AnalogVoltageScaler
from sklearn.pipeline import Pipeline
from dtw_classifier import DTWClassifier
from gesture_ensembler import HandGestureEnsembler
import time 
import sys

if __name__ == "__main__":
    from app_logger import setup_logging
    setup_logging() 

X_train, y_train = crt.getTrainData()

gyro_attr_names = ["cord_x", "cord_y", "cord_z"]
acc_attr_names = ["tilt_x", 'tilt_y', 'tilt_z']
fgr_attr_names = ['thumb', 'index', 'middle', 'ring', 'little']

all_pipelines = []
gyro_pipeline = Pipeline([       
        ('selector', DataFrameSelector(gyro_attr_names)),
        ('cord_norm', CoordinateNormalizer()),
        ('estimator', DTWClassifier(dist='euclidean', normalize=True)),
    ])
all_pipelines.append(gyro_pipeline)

acc_pipeline = Pipeline([       
        ('selector', DataFrameSelector(acc_attr_names)),
        ('acc_norm', AccelerometerNormalizer()),
        ('estimator', DTWClassifier(normalize=True)),
    ])
all_pipelines.append(acc_pipeline)

flex_pipeline = Pipeline([       
        ('selector', DataFrameSelector(fgr_attr_names)),
        ('std_scaler', AnalogVoltageScaler()),
        ('estimator', DTWClassifier(normalize=True)),
    ])
all_pipelines.append(flex_pipeline)

ensember = HandGestureEnsembler(all_pipelines)
ensember.fit(X_train, y_train)
X_test, y_test = crt.getTestData()

print(ensember.score(X_test, y_test))

