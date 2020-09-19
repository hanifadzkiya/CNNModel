import copy

from layers.base_layer import Layer
import numpy as np

from layers.pooling import Pooling


class MaxPooling(Pooling):
    def __init__(self, pool_size, stride):
        super(MaxPooling, self).__init__(pool_size, stride)

    def _find_pooling(self, matrix):
        return np.amax(matrix)

