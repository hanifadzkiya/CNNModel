from layers.pooling import Pooling
import numpy as np


class AvgPooling(Pooling):
    def __init__(self, pool_size, n_stride):
        super(AvgPooling, self).__init__(pool_size, n_stride)

    def _find_pooling(self, matrix):
        return np.average(matrix)

    def backward(self, din, lrate, momentum_rate):
        num_channels, orig_dim, *_ = self.last_input.shape      # gradients are passed through the indices of greatest
                                                                # value in the original pooling during the forward step

        dout = np.zeros(self.last_input.shape)                  # initialize derivative

        for c in range(num_channels):
            tmp_y = out_y = 0
            while tmp_y + self.pool_size[1] <= orig_dim:
                tmp_x = out_x = 0
                while tmp_x + self.pool_ize[0] <= orig_dim:
                    fill_value = din[out_x, out_y, c] / self.pool_size.size
                    dout[tmp_x:tmp_x + self.pool_size[0], tmp_y:tmp_y + self.pool_size[1], c]  = fill_value
                    tmp_x += self.n_stride
                    out_x += 1
                tmp_y += self.n_stride
                out_y += 1

        return dout
