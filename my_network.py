import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda


class DQNNetwork(Model):
    def __init__(self, n_features: int, n_actions: int):
        super(DQNNetwork, self).__init__()
        self.flatten = Flatten(input_shape=(n_features,))
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(128, activation='relu')
        self.d3 = Dense(n_actions, activation='linear')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
