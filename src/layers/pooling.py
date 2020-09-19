import copy

import numpy as np
from layers.base_layer import Layer


class Pooling(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input):
        matrix_output = []
        for matrix in input:
            size_matrix = (np.size(matrix, 0), np.size(matrix, 1))
            size_output = (size_matrix[0] // self.pool_size[0],
                           size_matrix[1] // self.pool_size[1])
            output = np.empty(size_output)
            for i in range(self.pool_size[0]):
                for j in range(self.pool_size[1]):
                    output[i][j] = self._find_pooling(matrix, (i * self.pool_size[0], j * self.pool_size[1]))

            matrix_output.append(output)

        return copy.deepcopy(np.array(matrix_output))

    def _find_pooling(self, matrix, loc):
        pass
