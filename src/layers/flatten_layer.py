import numpy as np
import copy
from layers.base_layer import Layer

class FlattenLayer(Layer):
    def forward(self, inputs):
        self.inputs = copy.deepcopy(inputs)
        return inputs.flatten()

    def backward(self, last_gradient, lrate, momentum_rate):
        return last_gradient.reshape(self.inputs.shape)