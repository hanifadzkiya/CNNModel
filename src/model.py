import copy

import numpy as np
import math
from layers.base_layer import Layer
from layers.activation import Activation


class Model:
    def __init__(self):
        self.layers = np.empty(0, dtype=Layer)

    def add(self, layer):
        self.layers = np.append(self.layers, layer)

    def forward(self, input):
        print(input.shape)
        self.output = copy.deepcopy(input)
        for i, layer in enumerate(self.layers):
            self.output = layer.forward(self.output)
            print("Input Shape After Layer : " + str(i) + " adalah " + str(self.output.shape))
        return self.output

    def backward(self, target, lrate, momentum_rate):
        print("target : " + str(target))
        last_layer = -(np.array(target) - self.output)
        print("last layer : " + str(last_layer))
        for i, layer in reversed(list(enumerate(self.layers))):
            last_layer = layer.backward(last_layer, lrate, momentum_rate)
            print("Last gradient shape for layer - " + str(i) + " adalah " + str(last_layer.shape))
 

    def fit(self, input,target,batch_size,epoch,learning_rate,momentum_rate):
        batch_count = math.ceil(len(input)/batch_size)
        
        for epoch_count in range(epoch):
            for i in range (batch_count):
                
                for j in range(batch_size):
                    temp = self.forward(input[i*batch_size+j])
                    error_gradient=0.5*(target[i*batch_size+j]-temp[0])
                    self.backward([target[i*batch_size+j]], learning_rate, momentum_rate)
                for layer in self.layers:
                    layer.reset_delta()
                for i, layer in reversed(list(enumerate(self.layers))):
                    layer.update_weight(learning_rate)

