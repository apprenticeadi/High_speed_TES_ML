import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils
from src.traces import Traces
from scipy.signal import find_peaks



TS_chi = np.loadtxt('params/chi_vals_TS.txt')[0:5]
ML_chi = np.loadtxt('params/chi_vals_ML.txt')
x_TS = np.arange(300,701,100)
x_ML = np.arange(200,1001,100)

plt.plot(x_ML, ML_chi, 'x', label = 'ML')
plt.plot(x_TS, TS_chi, '+', label = 'tail subtraction')

m,c = np.polyfit(x_ML[0:8], ML_chi[0:8],1 )

plt.plot(x_ML, x_ML*m+c,linestyle = 'dashed', label = 'ML fit excluding 1MHz')

plt.xlabel('frequency (kHz)')
plt.ylabel('Chi-Square value')
plt.legend()
plt.show()





