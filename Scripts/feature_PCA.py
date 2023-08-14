import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X = np.loadtxt('extracted_exp.txt')
y1 = np.loadtxt('labels.txt', delimiter = ',', unpack = True)
plt.scatter(y1,X[:,7] )
plt.show()

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)

# Determine the number of features in your dataset
num_features = X_train.shape[1]
# Select the top K features using mutual information
num_features_to_select = 3  # Adjust this number based on your actual feature count


selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
important_feature_ind = selector.get_support(indices=True)

important_features = X[:,[1,3,4]]
print(important_feature_ind)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter3D(important_features[:,0], important_features[:,1], important_features[:,2], c = y1, cmap = 'Set1')
ax.set_xlabel('A')
ax.set_ylabel('d')
ax.set_zlabel('e')

plt.show()
# Train a classifier (Random Forest in this case) on the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_selected, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)