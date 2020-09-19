from PIL import Image
import numpy as np
import time


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
        filter_layer=np.random.randint(-1,1,size=(3,3,3,3))
        # for i in range(self.n_filter):
        #     kernel=[]
        #     for j in range(self.filter_size[0]):
        #         temp=[]
        #         for k in range(self.filter_size[1]):
        #             temp.append(random.randint(-1, 1))
        #             kernel.append(temp)
        #     filter_layer.append(kernel)
        self.filter = filter_layer
        print(self.filter)


    def convolution(self, image_matrix):
        r,g,b    = image_matrix[:, :, 0], image_matrix[:, :, 1], image_matrix[:, :, 2]
        pad_length=self.padding
        width=self.inputs_size[0]+(2*pad_length)
        height=self.inputs_size[1]+(2*pad_length)
        r_backup=np.pad(r,pad_length,'constant')
        filter_width = self.filter_size[0]
        filter_height = self.filter_size[1]
        r_result=[]
        g_result=[]
        b_result=[]
        output_size = int((width - filter_width + (2*pad_length))/self.n_stride)+1
        result_matrix = np.zeros((output_size,output_size,3))
        for i in range(width-filter_width+1):
            r_temp=[]
            g_temp=[]
            b_temp=[]
            for j in range(height-filter_height+1):
                temp=[]
                for k in range(self.n_filter):
                    r_sum = np.sum(np.dot(r[j:j+filter_height,i:i+filter_width],self.filter[0][k]))
                    g_sum = np.sum(np.dot(g[j:j+filter_height,i:i+filter_width],self.filter[1][k]))
                    b_sum = np.sum(np.dot(b[j:j+filter_height,i:i+filter_width],self.filter[2][k]))
                    # print(i,j)
                    # result_matrix[i][j][0]+=r_sum
                    temp.append((r_sum+g_sum+b_sum))
                r_temp.append(temp[0])
                g_temp.append(temp[1])
                b_temp.append(temp[2])
            r_result.append(r_temp)
            g_result.append(g_temp)
            b_result.append(b_temp)
        print(len(r_result))
        print(r_result[0])
        print(len(g_result))
        print(len(b_result))


       
image = Image.open('cat.9.jpg')
arr= np.array(image)
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)
conv_layer= ConvolutionLayer((320,425),0,3,(3,3),1)
start = time.time()
print("hello")

conv_layer.convolution(arr)
end = time.time()
print(end - start)