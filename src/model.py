import copy

import numpy as np

from layers.base_layer import Layer
from layers.activation import Activation


class Model:
    def __init__(self):
        self.layers = np.empty(0, dtype=Layer)

    def add(self, layer):
        self.layers = np.append(self.layers, layer)

    def forward(self, input):
        self.output = copy.deepcopy(input)
        for i, layer in enumerate(self.layers):
            self.output = layer.forward(output)
            print("Input Shape After Layer : " + str(i) + " adalah " + str(output.shape))
        return self.output

    def backward(self, target):
        last_layer = target
        for i, layer in enumerate(self.layers):
            last_layer = layer.backward(last_layer)
