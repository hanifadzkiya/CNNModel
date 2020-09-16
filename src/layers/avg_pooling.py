from src.layers.pooling import Pooling
import numpy as np


class AvgPooling(Pooling):
    def __init__(self, pool_size):
        super(AvgPooling, self).__init__(pool_size)

    def _find_pooling(self, matrix, loc):
        result = 0
        for i in range(self.pool_size[0]):
            for j in range(self.pool_size[1]):
                result = result + matrix[loc[0] + i, loc[1] + j]

        return result / (self.pool_size[0] * self.pool_size[1])


input = []
input.append(np.array(np.array([[10, 20, 30, 0], [8, 12, 2, 0], [34, 70, 37, 4], [112, 100, 25, 12]])))
pooling = AvgPooling((2,2))
print(input)
print(pooling.compute(np.array(input)))

