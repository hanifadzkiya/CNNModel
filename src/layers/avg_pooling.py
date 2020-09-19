from layers.pooling import Pooling
import numpy as np


class AvgPooling(Pooling):
    def __init__(self, pool_size, n_stride):
        super(AvgPooling, self).__init__(pool_size, n_stride)

    def _find_pooling(self, matrix):
        return np.average(matrix)



