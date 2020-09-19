import copy

from src.layers.base_layer import Layer
import numpy as np

from src.layers.pooling import Pooling


class MaxPooling(Pooling):
    def __init__(self, pool_size):
        super(MaxPooling, self).__init__(pool_size)

    def _find_pooling(self, matrix, loc):
        result = 0
        for i in range(self.pool_size[0]):
            for j in range(self.pool_size[1]):
                result = max(result, matrix[loc[0] + i, loc[1] + j])

        return result


input = []
input.append(np.array(np.array([[10, 20, 30, 0], [8, 12, 2, 0], [34, 70, 37, 4], [112, 100, 25, 12]])))
pooling = MaxPooling((2,2))
print(input)
print(pooling.forward(np.array(input)))

