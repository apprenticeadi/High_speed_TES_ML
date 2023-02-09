import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split


frequency = 600

t_max = np.floor(5e4 / frequency) * 500

## TODO: make this if else statement less 'manual'
if frequency == 100:

    data_raw_ = np.loadtxt(r'../Data/all_traces_100kHz_middle.txt',
                           delimiter=',', unpack=True)
    data_raw = np.transpose(data_raw_)
elif frequency == 600:
    data_raw_ = np.loadtxt(r'../Data/all_traces_600kHz.txt',
                           delimiter=',', unpack=True)
    data_raw_ = np.transpose(data_raw_)

    '''  
    splitting the high frequency data
    '''
    idealSamples = 5e4 / frequency
    samples = np.floor(idealSamples) * 500
    period = int(idealSamples)

    ## TODO: Replace list operations with np array
    data_raw = []
    for data_set in data_raw_:
        for i in range(int(samples / idealSamples)):
            start = int(i * idealSamples)
            if start + period < samples:
                trace = data_set[start:start + period]
                data_raw.append(trace)
            else:
                pass
    data_raw = np.asarray(data_raw)
else:
    raise Exception('Code not good enough for this frequency ')


min_voltage = np.amin(data_raw)
max_voltage = np.amax(data_raw)
ymin = 5000 * (min_voltage // 5000)
ymax = 5000 * (max_voltage // 5000 + 1)


'''
take first x traces 
'''
num_traces = 100
data = data_raw[:num_traces, :t_max]  # This is the data we care about
plt.figure(f'{frequency}kHz traces raw')
for i in range(num_traces):
    plt.plot(data[i])
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim(0, t_max)
plt.ylim(ymin, ymax)

'''
To perform PCA, first zero the mean along each column
'''
col_means = np.mean(data, axis=0)
data_zeroed = data - col_means
plt.figure(f'{frequency}kHz traces with mean zeroed')
for i in range(num_traces):
    plt.plot(data_zeroed[i])
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim(0, t_max)

'''
Singular value decomposition to find factor scores and loading matrix
'''
P, Delta, QT = np.linalg.svd(data_zeroed, full_matrices=False)
F = P * Delta  # Factor scores
Q = QT.T  # Loading matrix

F2 = F ** 2
inertia_component = np.sum(F2, axis = 0)  # Sum of f squared for each principal component#
inertia_observation  = np.sum(F2, axis=1)  # Sum of all factor scores squared for each observation
total_inertia = np.sum(inertia_observation)
component_importance = inertia_component / total_inertia

'''
Truncate at first few principal components
'''
num_comp = 2
F_truncated = F[:, :num_comp]
data_cleaned = F_truncated @ QT[:num_comp, :] + col_means
plt.figure(f'{frequency}kHz traces truncated at {num_comp} principal components')
for i in range(num_traces):
    plt.plot(data_cleaned[i])
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim(0, t_max)
plt.ylim(ymin, ymax)

'''
Plot of loadings
'''
# This is definitely not the correlation plot shown in abdi&williams.
F2_truncated = F2[:, :num_comp]
loading = F2_truncated / inertia_observation[:, None]
plt.figure(f'Plot of each trace loading')
for i in range(num_traces):
    plt.plot(loading[i,0], loading[i,1], 'rx')
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'b-')
plt.xlabel('F1^2/d^2')
plt.ylabel('F2^2/d^2')
plt.axis('scaled')
plt.xlim(-2, 2)
plt.ylim(-2,2)
