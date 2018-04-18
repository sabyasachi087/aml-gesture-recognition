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

start_time = time.time()
for pl in all_pipelines:
    pl.fit(X_train, y_train)
print('Estimators are ready in %f seconds' % (time.time() - start_time))

X_test, y_test = crt.getTestData()
tst_idx = np.random.randint(0, len(y_test))

resultDf = pd.DataFrame(columns=['gesture', 'distance'])

print('Testing the gesture >>>', y_test[tst_idx], '<<<')
start_time = time.time()
for pl in all_pipelines:
    preds = pl.predict([X_test[tst_idx]])
    for gesture, distance in preds:        
        for idx in range(len(gesture)):
            resultDf.loc[len(resultDf)] = [gesture[idx], distance[idx]]

print('Prediction completed in %f seconds' % (time.time() - start_time))
resultDf['distance'] = resultDf['distance'].apply(lambda x : (1 - x) / len(resultDf))
resultDf = resultDf.groupby(['gesture'], as_index=False).sum()
total_dist = resultDf['distance'].sum()
resultDf['distance'] = resultDf['distance'].apply(lambda x : x / total_dist)
print(resultDf.ix[resultDf['distance'].idxmax()])
