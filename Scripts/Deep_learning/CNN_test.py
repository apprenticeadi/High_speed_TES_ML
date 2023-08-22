import tensorflow as tf
from src.utils import DataUtils, TraceUtils
import matplotlib.pyplot as plt
from src.ML_funcs import find_offset
import numpy as np
frequency = 700
multiplier = 1.5
power = 6

actual_data = DataUtils.read_high_freq_data(frequency,power = power,new = True)
shift = find_offset(frequency, power)
actual_data = actual_data - shift


best_model = tf.keras.models.load_model('models/700kHz_raw5.h5')

# Predict using the best model
predictions = best_model.predict(actual_data[..., np.newaxis])

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

y = np.bincount(predicted_labels)
x = range(len(y))
plt.bar(x,y)
plt.show()