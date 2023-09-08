import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
from src.ML_funcs import return_artifical_data

scale = False

dataset, true_labels = return_artifical_data(600,1.6,8)
num_samples = len(dataset)
data2 = dataset.copy()

max = np.max(data2.flatten())
range = max /18


if scale ==True:
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    range = 0.55


num_clusters = 11
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(dataset)


dbscan = DBSCAN(eps=range, min_samples=11)
dbscan_labels = dbscan.fit_predict(dataset)

kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
kmeans_nmi = normalized_mutual_info_score(true_labels, kmeans_labels)
kmeans_homogeneity = homogeneity_score(true_labels, kmeans_labels)

dbscan_ari = adjusted_rand_score(true_labels, dbscan_labels)
dbscan_nmi = normalized_mutual_info_score(true_labels, dbscan_labels)
dbscan_homogeneity = homogeneity_score(true_labels, dbscan_labels)


print("K-means:")
print(f"ARI: {kmeans_ari:.4f}")
print(f"NMI: {kmeans_nmi:.4f}")
print(f"Homogeneity: {kmeans_homogeneity:.4f}")

print('-'*30)
print("DBSCAN:")
print(f"ARI: {dbscan_ari:.4f}")
print(f"NMI: {dbscan_nmi:.4f}")
print(f"Homogeneity: {dbscan_homogeneity:.4f}")


