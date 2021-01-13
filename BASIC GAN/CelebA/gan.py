import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Gan:
    pre_processing = None
    generator_model = None
    discriminator_model = None

    def __init__(self):
        self.pre_processing = preprocessing.PreProcessor()
        self.generator_model = self.generator_model()
        self.discriminator_model = self.discriminator_model()

    def run(self):
        x_train, x_test, y_train, y_test = self.pre_processing.run()
        print()

    def generator_model(self):
        model = keras.Sequential(
            [
                layers.Dense(200, activation = 'relu', input_shape=(256, 256)),
                layers.Dense(3, activation="relu", name="layer2"),
                layers.Dense(4, name="layer3"),
            ]
        )
        print()
        return model
    def discriminator_model(self):
        model = None
        print()
        return model
