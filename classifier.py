import numpy as np
import pandas as pd
import curator as crt
from DataType import DataType
from dtw import fastdtw
from gesture_transformers import CoordinateNormalizer, AccelerometerNormalizer, DataFrameSelector, AnalogVoltageScaler
from sklearn.pipeline import Pipeline
from dtw_classifier import DTWClassifier
import time 
import sys

if __name__ == "__main__":
    from app_logger import setup_logging
    setup_logging() 

X_train, y_train = crt.getTrainData()

gyro_attr_names = ["cord_x", "cord_y", "cord_z"]
acc_attr_names = ["tilt_x", 'tilt_y', 'tilt_z']
fgr_attr_names = ['thumb', 'index', 'middle', 'ring', 'little']

gyro_pipeline = Pipeline([       
        ('selector', DataFrameSelector(gyro_attr_names)),
        ('cord_norm', CoordinateNormalizer()),
        ('estimator', DTWClassifier(dist='euclidean')),
    ])

acc_pipeline = Pipeline([       
        ('selector', DataFrameSelector(acc_attr_names)),
        ('acc_norm', AccelerometerNormalizer()),
        ('estimator', DTWClassifier()),
    ])

flex_pipeline = Pipeline([       
        ('selector', DataFrameSelector(fgr_attr_names)),
        ('std_scaler', AnalogVoltageScaler()),
        ('estimator', DTWClassifier()),
    ])

start_time = time.time()
gyro_predictor = gyro_pipeline.fit(X_train, y_train)
acc_predictor = acc_pipeline.fit(X_train, y_train)
flex_predictor = flex_pipeline.fit(X_train, y_train)
print('Estimators are ready in %f seconds' % (time.time() - start_time))

X_test, y_test = crt.getTestData()
tst_idx = np.random.randint(0, len(y_test))
print('Testing the gesture >>>', y_test[tst_idx], '<<<')

start_time = time.time()
print(gyro_predictor.predict([X_test[tst_idx]]))
print('Gyro prediction completed in %f seconds' % (time.time() - start_time))
start_time = time.time()
print(acc_predictor.predict([X_test[tst_idx]]))
print('Acc prediction completed in %f seconds' % (time.time() - start_time))
start_time = time.time()
print(flex_predictor.predict([X_test[tst_idx]]))
print('Flex prediction completed in %f seconds' % (time.time() - start_time))

