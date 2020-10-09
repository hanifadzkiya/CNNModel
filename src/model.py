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
        self.output = copy.deepcopy(input)
        for i, layer in enumerate(self.layers):
            self.output = layer.forward(self.output)
        return self.output

    def backward(self, target, lrate, momentum_rate):
        last_layer = -(np.array(target) - self.output)
        for i, layer in reversed(list(enumerate(self.layers))):
            last_layer = layer.backward(last_layer, lrate, momentum_rate)
 

    def fit(self, input,target,batch_size,epoch,learning_rate,momentum_rate):
        batch_count = math.ceil(len(input)/batch_size)
        
        for epoch_count in range(epoch):
            print("---------epoch : " + str(epoch_count) + "-------------")
            for i in range (batch_count):
                print("-------BATCH " + str(i) + "------------")
                for j in range(batch_size):
                    temp = self.forward(input[i*batch_size+j])
                    error_gradient=0.5*(target[i*batch_size+j]-temp[0])*(target[i*batch_size+j]-temp[0])
                    self.backward([target[i*batch_size+j]], learning_rate, momentum_rate)
                
                for i, layer in reversed(list(enumerate(self.layers))):
                    layer.update_weight(learning_rate)
                for layer in self.layers:
                    layer.reset_delta()
