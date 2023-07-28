import numpy as np
from src.utils import DataUtils
from src.traces import Traces
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

'''
attempt at k-means clustering
'''

'''create labelled dataset with synthetic data'''
multiplier = 3
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True

frequency = 200

data_100 = DataUtils.read_raw_data_new(100,0)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
labels = calibrationTraces.return_labelled_traces()
filtered_ind = np.where(labels == -1)[0]
filtered_traces = np.delete(data_100, filtered_ind, axis = 0)
filtered_label = np.delete(labels, filtered_ind)

print(str(100*((len(data_100) - len(filtered_label))/len(data_100))) +'% filtered')
filtered_data = Traces(100, filtered_traces)
data_high = filtered_data.overlap_to_high_freq(frequency)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data_high)

kmeans = KMeans(init = "random",n_clusters = 6,n_init = 10,max_iter = 300,random_state = 42)
dbscan = DBSCAN(eps=0.3)

kmeans.fit(scaled_data)
dbscan.fit(scaled_data)


print(kmeans.labels_[0:10], filtered_label[0:10])

# print(adjusted_rand_score(filtered_label, kmeans.labels_))
# print(adjusted_rand_score(filtered_label, dbscan.labels_))



