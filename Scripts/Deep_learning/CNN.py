import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
from scipy.optimize import curve_fit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.ML_funcs import return_artifical_data, find_offset
from src.utils import DataUtils
from src.traces import Traces

freq_values = np.arange(300, 301, 100)
power = 8
def poisson_norm(x, mu):
    return (mu ** x) * np.exp(-mu) / factorial(np.abs(x))
data100 = DataUtils.read_raw_data_new(100, power)
trace100 = Traces(100, data100, 1.8)
x, y = trace100.pn_bar_plot(plot=False)
fit, cov = curve_fit(poisson_norm, x, y / np.sum(y), p0=[3], maxfev=2000)
lam = fit[0]
chi_square = []

for frequency in freq_values:
    time_series, labels = return_artifical_data(frequency,1.6,power)

    signal_length = len(time_series[0])  # Length of each signal
    num_classes = len(np.bincount(labels))  # Number of classes

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
    model_checkpoint = ModelCheckpoint(filepath='models/300kHz_raw8.h5', save_best_only=True, save_weights_only=False)

    # Train the model
    history = model.fit(train_data[..., np.newaxis], train_labels_onehot, batch_size=50, epochs=250,
                        validation_data=(test_data[..., np.newaxis], test_labels_onehot),
                        callbacks=[reduce_lr, early_stop, model_checkpoint])

    # Load the best model
    best_model = tf.keras.models.load_model('models/300kHz_raw8.h5')

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

    actual_data = DataUtils.read_high_freq_data(frequency, power=power, new=True)
    shift = find_offset(frequency, power)
    actual_data = actual_data - shift

    actual_predictions = best_model.predict(actual_data[...,np.newaxis])
    actual_p = np.argmax(actual_predictions, axis=1)
    y = np.bincount(actual_p)/np.sum(np.bincount(actual_p))
    x = range(len(y))
    expected = poisson_norm(x, lam)

    chisq = []
    for i in range(len(expected)):
        chi = ((expected[i] - y[i]) ** 2) / expected[i]
        chisq.append((chi))
    chi_square.append(np.sum(chisq))

# np.savetxt('Chi_square_CNN.txt', chi_square)
# plt.plot(freq_values, chi_square)

