from PIL import Image
import numpy as np
import time
import sys
np.set_printoptions(threshold=sys.maxsize)

from layers.base_layer import Layer

class ConvolutionLayer(Layer):
    def __init__(self, padding, n_filter, filter_size, n_stride, inputs_size=None):
        self.inputs_size = inputs_size 
        self.padding = padding
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.filter=None
        self.n_stride = n_stride
        self.bias_filter=[]
        if inputs_size!=None:
            self.init_filter()
        self.input = None

    def init_filter(self):
        filter_layer = []
        filter_width = self.filter_size[0]
        filter_height = self.filter_size[1]
        filter_layer=np.random.randint(-1,2,size=(self.inputs_size[2],self.n_filter,filter_width,filter_height))
        self.bias_filter = np.zeros((self.n_filter),dtype=float)
        self.filter = filter_layer

    def forward(self, image_matrix):
        if self.filter is None:
            self.inputs_size = image_matrix.shape
            self.init_filter()

        # simpan buat backward
        self.input = image_matrix

        print(self.filter.shape)
        print(self.filter)


        width=self.inputs_size[0]
        height=self.inputs_size[1]
        pad_length=self.padding
        filter_width = self.filter_size[0]
        filter_height = self.filter_size[1]
        
        # calculate output size
        output_width = int((width - filter_width + (2*pad_length))/self.n_stride)+1
        output_height = int((height - filter_height + (2*pad_length))/self.n_stride)+1

        # padding
        width+=(2*pad_length)
        height+=(2*pad_length)

        for i in range(self.inputs_size[2]):
            image_matrix[:, :, i] = np.pad(image_matrix[:, :, i],pad_length,'constant')
        result_matrix = np.zeros((output_width,output_height,self.n_filter),dtype=float)

        for i in range(0,width-filter_width+1,self.n_stride):
            for j in range(0,height-filter_height+1,self.n_stride):
                for k in range(self.n_filter):
                    total_sum=0
                    for l in range(self.inputs_size[2]):                        
                        # print(np.sum(np.dot(image_matrix[:, :, l][j:j+filter_height,i:i+filter_width],self.filter[0][k])))
                        total_sum += np.sum(np.multiply(image_matrix[:, :, l][j:j+filter_height,i:i+filter_width],self.filter[l][k]))
                    # r_sum = np.sum(np.dot(r[j:j+filter_height,i:i+filter_width],self.filter[0][k]))
                    # g_sum = np.sum(np.dot(g[j:j+filter_height,i:i+filter_width],self.filter[1][k]))
                    # b_sum = np.sum(np.dot(b[j:j+filter_height,i:i+filter_width],self.filter[2][k]))
                    result_matrix[i,j,k] += total_sum
                    # add bias
                    result_matrix[i,j,k] += self.bias_filter[k]
        return result_matrix

    def backward(self, dl_dout=0, lrate=0.01, momentum_rate):
        dl_din = np.zeros(self.inputs_size)              # loss gradient of the input to the convolution operation
        dl_df = np.zeros(self.filter_size)               # loss gradient of filter

        idx_x = 0
        outx = 0
        while idx_x + self.filter_size[0] <= self.inputs_size[0]:
            idx_y = 0
            outy = 0
            while idx_y + self.filter_size[1] <= self.inputs_size[1]:
                for k in range(self.n_filter):
                    mtrx_area = self.input[idx_x:idx_x + self.filter_size[0], idx_y:idx_y + self.filter_size[1], :]
                    dl_df[:,k] += np.sum(dl_dout[outx, outy, :] * mtrx_area, axis=2)
                    dl_din[idx_x:idx_x + self.filter_size[0], idx_y:idx_y + self.filter_size[1], :] += dl_dout[outx, outy, :] * self.filter[:,k]
                idx_y += self.n_stride
                outy += 1
            idx_x += self.n_stride
            outx += 1

        #update filter
        self.filter -= lrate * dl_df
        return dl_din                                         # return the loss gradient for this layer's inputs
        