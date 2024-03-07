import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data_utils import DataParser, DataChopper
from src.traces import Traces

'''
script to produce PN distributions using tabular classifiers, specify power, whether FE and modeltype.
'''
#specify parameters
data_dir = 'RawData'
sub_power_name = 'raw_6'

test_size = 0.1
multiplier = 1.
num_bins = 1000
vertical_shift = True
triggered = True
pca_components = None

dataParser = DataParser(sub_dir=sub_power_name, parent_dir=data_dir)


data100 = dataParser.parse_data(100, interpolated=False, triggered=False)
refTraces = Traces(100, data100, multiplier, num_bins=num_bins)

# run pca on calibration traces if required.
if pca_components is None:
    calTraces = refTraces
else:
    # use pca cleaned-up traces as calibration traces.
    data_pca = refTraces.pca_cleanup(num_components=pca_components)
    calTraces = Traces(100, data_pca, multiplier, num_bins=num_bins)


frequency = 200

actual_data = dataParser.parse_data(frequency, interpolated=False,
                                    triggered=triggered)  # DataUtils.read_high_freq_data(frequency, power= power, new= True)
av_actual = np.mean(actual_data, axis=0)
period = actual_data.shape[1]
'''
Generate training data by overlapping the calibration traces 
'''
training_data, training_labels = calTraces.generate_training_data(frequency)  # untriggered.
av_training = np.mean(training_data, axis=0)
assert training_data.shape[1] == period

'''
Correct for vertical and horizontal shift
'''
h_offset = np.argmax(av_training) - np.argmax(av_actual)
if h_offset < 0:
    h_offset = h_offset + period

training_data, training_labels = DataChopper.chop_labelled_traces(training_data.flatten(), training_labels,
                                                                  period, trigger=h_offset)

if vertical_shift:
    v_offset = np.max(np.mean(training_data, axis=0)) - np.max(av_actual)
    training_data = training_data - v_offset


X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size=0.2)




param_grid = {
    'n_estimators': [200,300,400,500],
    'max_depth': [None, 1, 2, 3,4,5],
    'min_samples_split': [0,1,2,3,4],
    'min_samples_leaf': [0,1,2,3,4]
}

rf_model = RandomForestClassifier()
# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train a model with the best hyperparameters
best_rf_model = RandomForestClassifier(**best_params)
best_rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
