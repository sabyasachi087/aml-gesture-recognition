import numpy as np
import pandas as pd
import curator as crt
from DataType import DataType
from dtw import fastdtw
from gesture_transformers import CoordinateNormalizer

data_list = crt.getData(gesture='not_sure')
# x = data_list[0]
# y = data_list[1]
# print(fastdtw(x, y, 'euclidean'))
data = data_list[0]
X = data[["tilt_x","tilt_y","tilt_z"]]
norm = CoordinateNormalizer()
print(X)
print('************************')
print(norm.transform(X))