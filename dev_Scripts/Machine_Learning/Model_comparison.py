import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.ML_funcs import ML, return_artifical_data, extract_features
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
'''
script which compares the performances of different tabular classifiers
'''
power = 8
frequency = 600
feature_extraction = True
data, labels = return_artifical_data(frequency, 1.5, power)
peak_data = []

'''
calculate features for each time-series
'''
if feature_extraction == True:
    for series in tqdm(data):
        feature = extract_features(series)
        peak_data.append(feature)

    features = np.array(peak_data)
else:
    features = data

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
train_times = []
s1 = time.time()
svm_model = SVC()
svm_model.fit(X_train, y_train)
s2 = time.time()
train_times.append(s2-s1)


s1 = time.time()
rf_model = RandomForestClassifier(n_estimators=300)
rf_model.fit(X_train, y_train)
s2 = time.time()
train_times.append(s2-s1)

s1 = time.time()
boosted_model = GradientBoostingClassifier()
boosted_model.fit(X_train, y_train)
s2 = time.time()
train_times.append(s2-s1)

s1 = time.time()
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
s2 = time.time()
train_times.append(s2-s1)


models = [svm_model, rf_model, boosted_model, knn_model]
model_names = ['SVM', 'Random Forest', 'Boosted Decision Tree', 'K-Nearest Neighbors']

for model, name, build_time in zip(models, model_names,train_times):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Metrics for {name} Model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Build time: {build_time:.4f}")
    print("-" * 30)
