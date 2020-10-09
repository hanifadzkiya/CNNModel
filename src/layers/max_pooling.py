import copy

from layers.base_layer import Layer
import numpy as np

from layers.pooling import Pooling


class MaxPooling(Pooling):
    def __init__(self, pool_size, stride):
        super(MaxPooling, self).__init__(pool_size, stride)

    def _find_pooling(self, matrix):
        return np.amax(matrix)

    def backward(self, din, lrate, momentum_rate):
        print("shape", self.input.shape)
        widht, height, num_channels = self.input.shape      # gradients are passed through the indices of greatest
        
                                                    # value in the original pooling during the forward step
        dout = np.zeros(self.input.shape)                  # initialize derivative
        
        print("num_channels", num_channels)
        for c in range(num_channels):
            tmp_y = out_y = 0
            while tmp_y + self.pool_size[1] <= height:
                tmp_x = out_x = 0
                while tmp_x + self.pool_size[0] <= widht:
                    
                    # print(self.input.shape)
                    # print(tmp_x + self.pool_size[0])
                    # print(tmp_y + self.pool_size[1])
                    # print(c)
                    patch = self.input[tmp_x:tmp_x + self.pool_size[0], tmp_y:tmp_y + self.pool_size[1], c]    # obtain index of largest
                    (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)                     # value in patch
                    dout[tmp_y + x, tmp_x + y, c] += din[out_x, out_y, c]
                    tmp_x += self.n_stride
                    out_x += 1
                tmp_y += self.n_stride
                out_y += 1
        return dout

