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
        output = copy.deepcopy(input)
        for i, layer in enumerate(self.layers):
            # print("Run Layer : " + str(i))
            output = layer.forward(output)
        return output
model = Model()
model.add(Activation())
print(model.layers)
