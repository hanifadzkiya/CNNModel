from PIL import Image
import numpy as np
import random
import time
import sys
np.set_printoptions(threshold=sys.maxsize)

class ConvolutionLayer:
    def __init__(self, inputs_size, padding, n_filter, filter_size, n_stride):
        self.inputs_size = inputs_size
        self.padding = padding
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.filter=[]
        self.n_stride = n_stride
        self.init_filter()

    def init_filter(self):
        filter_layer = []
        filter_width = self.filter_size[0]
        filter_height = self.filter_size[1]
        filter_layer=np.random.randint(-1,2,size=(self.inputs_size[2],self.n_filter,filter_width,filter_height))
        self.filter = filter_layer

    def convolution(self, image_matrix):
        r,g,b    = image_matrix[:, :, 0], image_matrix[:, :, 1], image_matrix[:, :, 2]        
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

        r = np.pad(r,pad_length,'constant')
        g = np.pad(g,pad_length,'constant')
        b = np.pad(b,pad_length,'constant')

        result_matrix = np.zeros((output_width,output_height,self.n_filter),dtype=int)

        for i in range(0,width-filter_width+1,self.n_stride):
            for j in range(0,height-filter_height+1,self.n_stride):
                for k in range(self.n_filter):
                    r_sum = np.sum(np.dot(r[j:j+filter_height,i:i+filter_width],self.filter[0][k]))
                    g_sum = np.sum(np.dot(g[j:j+filter_height,i:i+filter_width],self.filter[1][k]))
                    b_sum = np.sum(np.dot(b[j:j+filter_height,i:i+filter_width],self.filter[2][k]))
                    total_sum = r_sum+g_sum+b_sum
                    result_matrix[i,j,k]+=total_sum
        return result_matrix
       
image = Image.open('cat.9.jpg')
# image = image.resize((200,200))
arr= np.array(image)
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)
conv_layer= ConvolutionLayer((320,425,3),1,3,(3,3),1)
start = time.time()
ans= conv_layer.convolution(arr)
print(ans[:,:,0])
print(ans[:,:,1])
print(ans[:,:,2])



end = time.time()
print(end - start)