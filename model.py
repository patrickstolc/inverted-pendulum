import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers
from keras.models import load_model


class Model:

    def __init__(self, input_layer, hidden_layer, input_activation, hidden_activation):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.init = initializers.RandomUniform(minval=-2, maxval=2)

        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        # create input layer
        self.model.add(Dense(self.input_layer["output_size"], input_dim=self.input_layer["input_size"], kernel_initializer=self.init))
        self.model.add(Activation(self.input_activation))
        # create hidden layer(s)
        for l in self.hidden_layer:
            self.model.add(Dense(l, kernel_initializer=self.init))
            self.model.add(Activation(self.hidden_activation))

        # output layer
        self.model.add(Dense(1, kernel_initializer=self.init))
        self.model.add(Activation('linear'))

        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adadelta')

    def get_weights(self):
        w = []
        for layer in self.model.layers:
            w.append(layer.get_weights())

        return w

    def set_weights(self, w):
        for i, layer in enumerate(self.model.layers):
            self.model.layers[i].set_weights(w[i])

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path + '_model.h5')
        self.model.save_weights(path + '_weights.h5')
