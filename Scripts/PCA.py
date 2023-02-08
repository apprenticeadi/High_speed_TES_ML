import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

data_100_ = np.loadtxt(r'../Data/all_traces_100kHz_middle.txt',
                       delimiter=',', unpack=True)
data_100 = np.transpose(data_100_)


min_voltage = np.amin(data_100)
max_voltage = np.amax(data_100)
ymin = 5000 * (min_voltage // 5000)
ymax = 5000 * (max_voltage // 5000 + 1)

'''
plot first x traces in 100kHZ data
'''
num_traces = 100
plt.figure('100kHz traces')
for i in range(num_traces):
    plt.plot(data_100[i][:200])
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim(0, 200)
plt.ylim(ymin, ymax)

'''
Perform PCA on the data
'''