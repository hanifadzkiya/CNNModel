import copy
import numpy as np
from layers.base_layer import Layer

class Activation(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(input, 0)

    def backward(self, last_gradient, lrate, momentum_rate):
        
        input_flat = self.input.flatten()
        last_flat = last_gradient.flatten()

        for x, inputs in enumerate(input_flat):
            input_flat[x] = 0 if last_flat[x] < 0 else last_flat[x]

        return input_flat.reshape(self.input.shape)