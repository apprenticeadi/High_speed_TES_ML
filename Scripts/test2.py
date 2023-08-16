import numpy as np
import matplotlib.pyplot as plt
from src.ML_funcs import return_artifical_data
from src.utils import DataUtils
from src.traces import Traces


'''
comparison of offset subtraction methods
'''
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
freq_values = np.arange(200,1001,100)
differences = []
for freq,ax in zip(freq_values, axs.ravel()):
    actual_data = DataUtils.read_high_freq_data(freq,5,new = True)
    art_data,label = return_artifical_data(freq,1.8,5)
    trace1 = Traces(freq,actual_data,1.8)
    art_trace = Traces(freq, art_data, 1.8)
    av1 ,a,b = trace1.average_trace(plot = False)
    av2,c,d = art_trace.average_trace(plot = False)
    diff = np.max(av1) - np.max(av2)
    differences.append(diff)
    ax.plot(av1, label = 'data')
    ax.plot(av2, label = 'artificial')
    # plt.plot(freq,off,'+')
    # plt.plot(freq,diff,'o')
    ax.set_title(freq)
    ax.legend()
print(list(differences))
plt.show()
#
#[1215.3951373568752, 1101.6401777195333, 909.7344144653093, 966.391281935189, 1088.2631153452876, 1205.8939480836534, 1360.1713700452046, 1852.4754714202963, 2261.182167195711]
# data = DataUtils.read_high_freq_data(600,5,new=True)
# trace = Traces(600,data,1.5)
# trace.fit_histogram(plot = True)
# plt.show()

# data100 = DataUtils.read_raw_data_new(100,5)
# trace = Traces(100,data100,1.5)
# off,_ = trace.subtract_offset()
# print(off)
