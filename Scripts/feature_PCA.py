import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming 'features' is your numpy array containing feature vectors and 'labels' is the corresponding labels array
X = np.loadtxt('extracted_params_exp.txt')
y = np.loadtxt('labels.txt', delimiter = ',', unpack = True)
print(X[0], X[1])

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming X contains your feature vectors and y contains the corresponding labels
# X.shape should be (20000, num_features)
# y.shape should be (20000,)

# Encode labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Determine the number of features in your dataset
num_features = X_train.shape[1]

# Select the top K features using mutual information
num_features_to_select = 100  # Adjust this number based on your actual feature count
if num_features_to_select > num_features:
    num_features_to_select = num_features

selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train a classifier (Random Forest in this case) on the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_selected, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)