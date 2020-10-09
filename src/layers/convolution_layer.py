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
        self.dl_df = None
        self.n_stride = n_stride
        self.bias_filter=[]
        self.dl_db = np.zeros(n_filter)
        if inputs_size!=None:
            self.init_filter()
        self.input = None

    def init_filter(self):
        filter_layer = []
        filter_width = self.filter_size[0]
        filter_height = self.filter_size[1]
        filter_layer=np.random.uniform(low=-1.0,high=1.01,size=(self.inputs_size[2],self.n_filter,filter_width,filter_height))

        self.bias_filter = np.zeros((self.n_filter),dtype=float)
        self.filter = filter_layer

    def forward(self, image_matrix):
        if self.filter is None:
            self.inputs_size = image_matrix.shape
            self.init_filter()

        # simpan buat backward
        self.input = image_matrix

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
                        total_sum += np.sum(np.multiply(image_matrix[:, :, l][j:j+filter_height,i:i+filter_width],self.filter[l][k]))
                    result_matrix[i,j,k] += total_sum
                    # add bias
                    result_matrix[i,j,k] += self.bias_filter[k]
        self.output = result_matrix
        return result_matrix
    
    def reset_delta(self):
        self.dl_df = np.zeros((self.dl_df.shape),dtype=float)

    def update_weight(self, lrate):
        self.filter -= lrate * self.dl_df

    def backward(self, dl_dout=0, lrate=0.01, momentum_rate=0.1):
        

        dl_din = np.zeros(self.inputs_size)              # loss gradient of the input to the convolution operation

        idx_x = 0
        outx = 0
        
        width=self.inputs_size[0]
        height=self.inputs_size[1]
        pad_length=self.padding
        filter_width = dl_dout.shape[0]
        filter_height = dl_dout.shape[1]
        # calculate output size
        output_width = int((width - filter_width + (2*pad_length))/self.n_stride)+1
        output_height = int((height - filter_height + (2*pad_length))/self.n_stride)+1
        if self.dl_df is None:
            self.dl_df = np.zeros((self.inputs_size[2], dl_dout.shape[2], output_width, output_height),dtype=float)
        # padding
        width+=(2*pad_length)
        height+=(2*pad_length)
        for channel in range(dl_dout.shape[2]):
            for i in range(0,width-filter_width+1,self.n_stride):
                for j in range(0,height-filter_height+1,self.n_stride):
                    for k in range(self.n_filter):
                        total_sum=0
                        for l in range(self.inputs_size[2]):                        
                            total_sum += np.sum(np.multiply(self.input[:, :, l][j:j+filter_height,i:i+filter_width],dl_dout[channel][l][k]))
                        self.dl_df[channel,k,i,j] += total_sum
        return dl_din
