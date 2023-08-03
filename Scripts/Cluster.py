import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils
from src.traces import Traces
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from src.ML_funcs import return_artifical_data
'''
attempt at k-means clustering
'''

'''create labelled dataset with synthetic data'''
frequency = 200
data_high, filtered_label = return_artifical_data(frequency,3)
scaler = StandardScaler()

#scaled_data = scaler.fit_transform(data_high)

kmeans = KMeans(init = "random",n_clusters = 6,n_init = 10,max_iter = 300,random_state = 42)
dbscan = DBSCAN(eps=0.3)

dbscan.fit(data_high)

actual_data = DataUtils.read_high_freq_data(frequency, power = 2, new = True, trigger = False)

predictions = dbscan.predict(actual_data)

counts = np.bincount(predictions)

plt.bar(list(range(len(counts))), counts)
plt.show()



