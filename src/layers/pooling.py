import copy

import numpy as np
from layers.base_layer import Layer


class Pooling(Layer):
    def __init__(self, pool_size, n_stride):
        self.pool_size = pool_size
        self.n_stride = n_stride

    def forward(self, input):
        width = input.shape[0]
        height = input.shape[1]
        size_output = (((width - self.pool_size[0]) // self.n_stride)+1,
                           ((height - self.pool_size[1]) // self.n_stride)+1,
                           input.shape[2])
        matrix_output = np.empty(size_output)
        num_filter = input.shape[2]
        for i in range(num_filter):
            matrix = input[:, :, i]
            # size_matrix = (np.size(matrix, 0), np.size(matrix, 1))
            
            output = np.empty(size_output)
            # width = size_matrix[1]
            # height = size_matrix[0]
            for j in range(size_output[0]):
                for k in range(size_output[1]):
                    # print(str(i) + ' ' + str(j))
                    pos = (self.n_stride * j, self.n_stride * k)
                    # output[j][k] = self._find_pooling(matrix[pos[0]:pos[0] + self.pool_size[0], pos[1]:pos[1] + self.pool_size[1]])
                    matrix_output[j,k,i] = self._find_pooling(matrix[pos[0]:pos[0] + self.pool_size[0], pos[1]:pos[1] + self.pool_size[1]])


            # matrix_output.append(output)
        # print(matrix_output.shape)
        return copy.deepcopy(np.array(matrix_output))

    def _find_pooling(self, matrix, loc):
        pass
