import copy

import numpy as np
from layers.base_layer import Layer


class Pooling(Layer):
    def __init__(self, pool_size, n_stride):
        self.pool_size = pool_size
        self.n_stride = n_stride
        self.input = []

    def forward(self, input):
        self.input = copy.deepcopy(input)
        width = input.shape[0]
        height = input.shape[1]
        size_output = (((width - self.pool_size[0]) // self.n_stride)+1,
                           ((height - self.pool_size[1]) // self.n_stride)+1,
                           input.shape[2])
        matrix_output = np.empty(size_output)
        num_filter = input.shape[2]
        for i in range(num_filter):
            matrix = input[:, :, i]
            
            output = np.empty(size_output)

            for j in range(size_output[0]):
                for k in range(size_output[1]):
                    pos = (self.n_stride * j, self.n_stride * k)
                    matrix_output[j,k,i] = self._find_pooling(matrix[pos[0]:pos[0] + self.pool_size[0], pos[1]:pos[1] + self.pool_size[1]])



        return copy.deepcopy(np.array(matrix_output))

