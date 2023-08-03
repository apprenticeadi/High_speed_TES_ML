import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from src.utils import DataUtils
from src.traces import Traces
import tensorflow as tf
import numpy as np
from src.ML_funcs import return_artifical_data
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical


time_series, labels = return_artifical_data(600,3)
# Initialize k-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize list to store Test Accuracies
test_accs = []

# Initialize list to store Test Losses
test_losses = []

X_train, X_test, y_train, y_test = train_test_split(time_series, labels, test_size=0.2, random_state=42)

# Convert labels to categorical format
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print(time_series.shape)

input_shape = (time_series.shape[1], 1)  # assuming the shape is (length, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

length = len(X_train)
# Define the CNN model
model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dense(num_classes, activation='softmax'))
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

'''
model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print("Test Accuracy:", accuracy)
'''





# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=8, epochs=13, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print("Test Accuracy:", accuracy)

actual_data = DataUtils.read_high_freq_data(600,2, new = True)
targetTraces = Traces(frequency=600, data=actual_data, multiplier=3, num_bins=1000)
offset_target = 0.5*targetTraces.return_av_diff()
actual_data = actual_data - offset_target
actual_data = np.reshape(actual_data, (actual_data.shape[0], actual_data.shape[1], 1))
predictions = model.predict(actual_data)
predicted_class = np.argmax(predictions, axis = 1)

x_vals = list(range(len(np.bincount(predicted_class))))
y_vals = np.bincount(predicted_class)

plt.bar(x_vals, y_vals)
plt.show()