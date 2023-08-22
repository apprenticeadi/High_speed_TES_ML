import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.ML_funcs import return_artifical_data
time_series, labels = return_artifical_data(600,1.6,8)

signal_length = len(time_series[0])  # Length of each signal
num_classes = 11  # Number of classes

# Preprocessing: Split data and one-hot encode labels
train_data, test_data, train_labels, test_labels = train_test_split(time_series, labels, test_size=0.2, random_state=42)
train_labels_onehot = to_categorical(train_labels, num_classes=num_classes)
test_labels_onehot = to_categorical(test_labels, num_classes=num_classes)

# Build the CNN architecture
model = Sequential()

model.add(Conv1D(filters=16, kernel_size=11, padding='same', strides=1, input_shape=(signal_length, 1)))
model.add(BatchNormalization())
model.add(ReLU())

model.add(Conv1D(filters=16, kernel_size=11, padding='same', strides=1))
model.add(BatchNormalization())
model.add(ReLU())

model.add(GlobalAveragePooling1D())

model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-4)
early_stop = EarlyStopping(patience=40)
model_checkpoint = ModelCheckpoint(filepath='models/best_model.h5', save_best_only=True, save_weights_only=False)

# Train the model
history = model.fit(train_data[..., np.newaxis], train_labels_onehot, batch_size=50, epochs=250,
                    validation_data=(test_data[..., np.newaxis], test_labels_onehot),
                    callbacks=[reduce_lr, early_stop, model_checkpoint])

# Load the best model
best_model = tf.keras.models.load_model('models/best_model.h5')

# Predict using the best model
predictions = best_model.predict(test_data[..., np.newaxis])

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
# Calculate precision, recall, F1-score, and accuracy
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(test_labels, predicted_labels)

# Print evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)