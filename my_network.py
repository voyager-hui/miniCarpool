import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda


class DQNNetwork(Model):
    def __init__(self, n_features: int, n_actions: int):
        super(DQNNetwork, self).__init__()
        self.flatten = Flatten(input_shape=(n_features,))
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(128, activation='relu')
        self.d3 = Dense(n_actions, activation='softmax')

    def call(self, x):
        x1 = self.flatten(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        return self.d3(x3)
