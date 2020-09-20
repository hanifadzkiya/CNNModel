import numpy as np
from layers.base_layer import Layer

class FlattenLayer(Layer):
    def forward(self, inputs):
        return inputs.flatten()
