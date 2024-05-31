import numpy as np
import os
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from tes_resolver.classifier import Classifier
from tes_resolver.traces import Traces
import tes_resolver.config as config

class CNNClassifier(Classifier):

    def __init__(self, test_size=0.25):

        self.test_size = test_size
        self.default_dir = os.path.join(config.home_dir, 'saved_classifiers', f'CNNClassifier')

        self._params = {
            'accuracy_score': 0.,
            'precision': 0.,
            'recall': 0.,
            'f1-score': 0.,
            'time_stamp': config.time_stamp,
        }

        self.classifier = None

    def train(self, trainingTraces: Traces, max_epoch = 250, checkpoint_file=None):
        # Training data
        data = trainingTraces.data
        labels = trainingTraces.labels
        signal_length = data.shape[1]
        num_classes = len(set(labels))

        # Train test split
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=self.test_size)
        train_labels_onehot = to_categorical(train_labels, num_classes=num_classes)
        test_labels_onehot = to_categorical(test_labels, num_classes=num_classes)

        # create model
        model = Sequential()

        model.add(Conv1D(filters=16, kernel_size=20, padding='same', strides=1, input_shape=(signal_length, 1)))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Conv1D(filters=8, kernel_size=10, padding='valid', strides=1))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(GlobalAveragePooling1D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])


        # train model
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-4)
        early_stop = EarlyStopping(patience=40)

        if checkpoint_file is None:
            checkpoint_dir = os.path.join(self.default_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, f'models_checkpoint_{config.time_stamp}.h5')

        model_checkpoint = ModelCheckpoint(filepath=checkpoint_file, save_best_only=True, save_weights_only=False)

        history = model.fit(train_data[..., np.newaxis], train_labels_onehot, batch_size=20, epochs=max_epoch,
                            validation_data=(test_data[..., np.newaxis], test_labels_onehot),
                            callbacks=[reduce_lr, early_stop, model_checkpoint])

        # load highest performing model
        best_model = tf.keras.models.load_model(checkpoint_file)
        self.classifier = best_model

        # predict test data
        predictions = best_model.predict(test_data[..., np.newaxis])
        predicted_labels = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(test_labels, predicted_labels)

        self._params.update({
            'accuracy_score': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
        })


    def predict(self, unknownTraces: Traces, update=False):
        actual_data = unknownTraces.data

        predictions = self.classifier.predict(actual_data[...,np.newaxis])
        labels = np.argmax(predictions, axis=1)

        if update:
            unknownTraces.labels = labels
        return labels

    def save(self, filename=None, filedir=None):
        pass

    def load(self, filename, filedir=None):
        pass
