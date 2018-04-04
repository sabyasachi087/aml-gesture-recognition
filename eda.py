from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import curator as crt
from DataType import DataType

data_list = crt.getData(gesture='correct')
data = data_list[1]
X = np.array(data.tilt_x.astype('float64').values).astype(int)
Y = np.array(data.tilt_y.astype('float64').values).astype(int)
Z = np.array(data.tilt_z.astype('float64').values).astype(int)
print(data)

# Plotting graph
data_0 = data_list[0]
acc_ts = data[["tilt_x","tilt_y","tilt_z"]].cumsum()
acc_ts_0 = data_0[["tilt_x","tilt_y","tilt_z"]].cumsum()
acc_ts.plot()
acc_ts_0.plot()
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# 
# ax.scatter(X, Y, Z)
# ax.legend()
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
plt.legend()
plt.show()
