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
from tsfresh import extract_features
if __name__ == '__main__':
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


    data_dict = {'time_series': data_high.tolist(), 'label': filtered_label}
    data_df = pd.DataFrame.from_dict(data_dict)
    data_df['id'] = data_df.index

    time_series_list = data_df.apply(lambda row: row['time_series'], axis=1).tolist()
    data_df = pd.DataFrame({'id:df['id})



    data_df['time_series'] = data_df['time_series'].apply(pd.series)


    print('beginning feature extraction')
    extracted_features = extract_features(data_df, column_id='label', column_sort=None)
    print('feature extraction complete')

    # Step 2: Split the data into training and testing sets
    X = extracted_features.drop('label', axis=1).values
    y = extracted_features['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train a classifier (you can choose any other classifier if you prefer)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Step 4: Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Step 5: Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Step 6: Get the feature importances from the trained classifier
    feature_importances = clf.feature_importances_

    # Step 7: Sort features based on importance
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = extracted_features.drop('label', axis=1).columns[sorted_indices]

    # Print the sorted features and their corresponding importances
    print("Top features and their importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(f"{feature}: {importance}")



