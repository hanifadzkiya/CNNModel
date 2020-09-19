import copy
import numpy as np

from src.layers.base_layer import Layer


class Activation(Layer):
    def forward(self, input):
        result = copy.deepcopy(input)

        for i in range(np.size(result, 0)):
            for j in range(np.size(result, 1)):
                result[i][j] = max(0, result[i][j])

        return copy.deepcopy(result)