import copy
import numpy as np

from layers.base_layer import Layer

class Activation(Layer):
    def forward(self, input):
        return np.maximum(input, 0)
