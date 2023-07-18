import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import DataUtils
from src.traces import Traces
import seaborn as sns
import pandas as pd

multiplier = 3
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True

data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
_ = calibrationTraces.subtract_offset()
labels = calibrationTraces.return_labelled_traces()
filtered_ind = np.where(labels == -1)[0]
filtered_traces = np.delete(data_100, filtered_ind, axis = 0)
filtered_label = np.delete(labels, filtered_ind)

frequency = 500
filtered_data = Traces(100, filtered_traces)
data_high = filtered_data.overlap_to_high_freq(frequency)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(filtered_traces, filtered_label, test_size=0.2, random_state=42)

# Define a custom transformer to scale the 1D time series data
class TimeSeriesScaler(StandardScaler):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([self._scale_series(series) for series in X])

    def _scale_series(self, series):
        return (series - np.mean(series)) / np.std(series)

# Perform feature selection using SelectKBest with ANOVA (f_classif) score as the criterion
k_best = 10  # Select the top features
feature_selector = SelectKBest(score_func=f_classif, k=k_best)

X_train_scaled = TimeSeriesScaler().fit_transform(X_train)
X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)


X_test_scaled = TimeSeriesScaler().fit_transform(X_test)
X_test_selected = feature_selector.transform(X_test_scaled)

# Get the selected feature indices
selected_feature_indices = feature_selector.get_support(indices=True)


# Train a Random Forest classifier using the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_selected, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_selected)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Selected Feature Indices:", selected_feature_indices)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", conf_matrix)


X_train_selected_df = pd.DataFrame(X_train_selected)

# Compute the correlation matrix
corr_matrix = X_train_selected_df.corr()

# Create a correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Selected Features")
plt.show()

