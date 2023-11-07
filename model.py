import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input


class model:
    def __init__(self, path):
        self.model_ResNet50_1 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50.hdf5'))
        self.model_ResNet50_2 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (1).hdf5'))
        self.model_ResNet50_3 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (2).hdf5'))
        self.model_ResNet50_4 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (3).hdf5'))
        self.model_ResNet50_5 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (4).hdf5'))
        self.model_ResNet50_6 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (5).hdf5'))
        self.model_ResNet50_7 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (6).hdf5'))
        self.model_ResNet50_8 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (7).hdf5'))
        self.model_ResNet50_9 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (8).hdf5'))
        self.model_ResNet50_10 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (9).hdf5'))
        self.model_ResNet50_11 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50 (10).hdf5'))
        self.model_ResNet50_12 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50_1.hdf5'))
        self.model_ResNet50_13 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet50_2.hdf5'))
        self.model_ResNet152 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet152.hdf5'))
        self.model_ResNet152_1 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet152_1.hdf5'))
        self.model_ResNet152_2 = tf.keras.models.load_model(os.path.join(path, 'model_ResNet152_2.hdf5'))
       

        self.models = []
        self.models.append(self.model_ResNet152)
        self.models.append(self.model_ResNet152_1)
        self.models.append(self.model_ResNet152_2)
        self.models.append(self.model_ResNet50_1)
        self.models.append(self.model_ResNet50_2)
        self.models.append(self.model_ResNet50_3)
        self.models.append(self.model_ResNet50_4)
        self.models.append(self.model_ResNet50_5)
        self.models.append(self.model_ResNet50_6)
        self.models.append(self.model_ResNet50_7)
        self.models.append(self.model_ResNet50_8)
        self.models.append(self.model_ResNet50_9)
        self.models.append(self.model_ResNet50_10)
        self.models.append(self.model_ResNet50_11)
        self.models.append(self.model_ResNet50_12)
        self.models.append(self.model_ResNet50_13)
    def predict(self, X):
        # Insert your preprocessing here
        X = preprocess_input(X)
        yhats = [model.predict(X) for model in self.models]
        yhats = np.array(yhats)
        # sum across ensemble members
        summed = np.sum(yhats, axis=0)
        # argmax across classes
        out = tf.argmax(summed, axis=-1)
        return out