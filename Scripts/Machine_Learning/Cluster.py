import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
from src.ML_funcs import return_artifical_data
import time

scale = False
dataset, true_labels = return_artifical_data(600,1.6,8)
num_samples = len(dataset)
data2 = dataset.copy()

max = np.max(data2.flatten())
range = max /18
# Preprocess the data
if scale ==True:
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    range = 0.55

# Perform K-means clustering
cluster1 = time.time()
num_clusters = 11
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(dataset)
cluster2 = time.time()
kmeanstime = cluster2-cluster1

# Perform DBSCAN clustering
db1 = time.time()
dbscan = DBSCAN(eps=range, min_samples=11)
dbscan_labels = dbscan.fit_predict(dataset)
db2 = time.time()
dbtime = db2 - db1


# Calculate clustering evaluation metrics
kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
kmeans_nmi = normalized_mutual_info_score(true_labels, kmeans_labels)
kmeans_homogeneity = homogeneity_score(true_labels, kmeans_labels)

dbscan_ari = adjusted_rand_score(true_labels, dbscan_labels)
dbscan_nmi = normalized_mutual_info_score(true_labels, dbscan_labels)
dbscan_homogeneity = homogeneity_score(true_labels, dbscan_labels)

# Print evaluation metrics
print("K-means:")
print(f"ARI: {kmeans_ari:.4f}")
print(f"NMI: {kmeans_nmi:.4f}")
print(f"Homogeneity: {kmeans_homogeneity:.4f}")
print(f"build time: {kmeanstime:.4f}")
print('-'*30)
print("DBSCAN:")
print(f"ARI: {dbscan_ari:.4f}")
print(f"NMI: {dbscan_nmi:.4f}")
print(f"Homogeneity: {dbscan_homogeneity:.4f}")
print(f"build time: {dbtime:.4f}")

