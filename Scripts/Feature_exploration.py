import numpy as np
import matplotlib.pyplot as plt
from src.ML_funcs import ML, return_artifical_data, extract_features, find_offset
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from src.utils import DataUtils
from src.traces import Traces

'''
script containing various functions that are useful for ML:
1. feature_bar_plots returns the scores of each feature in the feature extraction
2. all_feature_plot returns a subplot of the 3d feature-space plot of the 'top 3 features'
3. PCA_test performs a pca decomposition on the testing data, the performs it on the actual data and looks for variance
4. one_feature_plot is the same as 2 but for one rep rate
5. art_trace_comp plots the average trace for both the artificial and real data for a given power and reprate
'''
frequency = 800
power = 7
feature_names =  ["peak_loc", "average", "std", "energy", "freq", "max_peak", "rise_time", "crest", "kurt", "area"]


freq_values = np.arange(200,1001,100)


def feature_bar_plots(freq_values, power):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    for frequency,ax in zip(freq_values, axs.ravel()):

        data, labels = return_artifical_data(frequency=frequency, multiplier=1.8, power = power)

        features = []

        for series in data:
            extracted_features = extract_features(series)
            features.append(extracted_features)

        features = np.array(features)

        select = SelectKBest(score_func=f_classif, k=3)
        best = select.fit(features, labels)
        f_scores = select.scores_
        feature_index = select.get_support(indices=True)

        print(feature_index)
        print(f'most important features : {feature_names[feature_index[0]]}, {feature_names[feature_index[1]]}, {feature_names[feature_index[2]]}')
        x = range(len(f_scores))

        ax.bar(x, f_scores)
        ax.set_ylabel('score')
        ax.set_xticks(x)
        ax.set_xticklabels(labels = feature_names, rotation = 35)
        ax.set_title(f'scores for {frequency}kHz, at power: {power}')

    plt.tight_layout()
    plt.show()

def all_feature_plots(freq_values, power):
    fig_3d, axs_3d = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), subplot_kw={'projection': '3d'})

    for i, frequency, ax_3d in zip(range(9), freq_values, axs_3d.ravel()):
        data, labels = return_artifical_data(frequency=frequency, multiplier=1.8, power = power)

        features = []

        for series in data:
            extracted_features = extract_features(series)
            features.append(extracted_features)

        features = np.array(features)
        max_peak, area , energy = features[:,5], features[:,9], features[:,3]
        sc = ax_3d.scatter(max_peak, area, energy, c=labels, cmap='viridis')
        ax_3d.set_xlabel('Max Peak')
        ax_3d.set_ylabel('Area')
        ax_3d.set_zlabel('Energy')
        ax_3d.set_title(f'{frequency} kHz, Power: {power}')


    cbar = plt.colorbar(sc, ax=axs_3d, pad=0.1)
    cbar.set_label('Label')

    plt.tight_layout()
    plt.show()

def PCA_test(frequency, power):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    data, labels = return_artifical_data(frequency, 1.8, power)

    actual_data = DataUtils.read_high_freq_data(frequency, power=power, new=True)
    shift = find_offset(frequency, power)
    actual_data = actual_data - shift

    pca = PCA(n_components=3, random_state=0)

    pca.fit(actual_data, labels)

    embedded = pca.transform(actual_data)

    ax.scatter3D(embedded[:, 0], embedded[:, 1], embedded[:, 2])
    plt.show()

def one_feature_plot(frequency, power):
    ax = plt.axes(projection = '3d')
    data, labels = return_artifical_data(frequency=frequency, multiplier=1.8, power=power)

    features = []

    for series in data:
        extracted_features = extract_features(series)
        features.append(extracted_features)

    features = np.array(features)
    max_peak, area, energy = features[:, 5], features[:, 9], features[:, 3]
    sc = ax.scatter(max_peak, area, energy, c=labels)
    ax.set_xlabel('Max Peak')
    ax.set_ylabel('Area')
    ax.set_zlabel('Energy')
    ax.set_title(f'{frequency} kHz, Power: {power}')
    plt.show()


def art_trace_comp(frequency, power):
    data, labels = return_artifical_data(frequency=frequency, multiplier=1.8, power=power)
    actual_data = DataUtils.read_high_freq_data(frequency, power=power, new=True)
    shift = find_offset(frequency, power)
    actual_data = actual_data - shift

    art_traces = Traces(frequency=frequency,data = data,  multiplier=1.8)
    actual_traces = Traces(frequency=frequency,data = actual_data,  multiplier=1.8)

    average,p1,p2 = art_traces.average_trace(plot = False)
    real_average, f1,f2 = actual_traces.average_trace(plot=False)
    plt.plot(average, label = 'artificial data')
    plt.plot(real_average, label = 'real data')
    plt.legend()
    plt.show()