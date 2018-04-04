from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import curator as crt
from DataType import DataType

data_list = crt.getData(gesture='bye')
data = data_list[1]
X = np.array(data.cord_x.astype('float64').values).astype(int)
Y = np.array(data.cord_y.astype('float64').values).astype(int)
Z = np.array(data.cord_z.astype('float64').values).astype(int)
print(data)

# Plotting graph


fig = plt.figure()
ax = fig.gca(projection='3d')


ax.plot(X, Y, Z, label='parametric curve')
ax.legend()

plt.show()
