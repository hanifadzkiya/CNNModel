import numpy as np

from src.layers.base_layer import Layer


def relu(inputs):
    for i in range(len(inputs)):
        inputs[i] = 0 if inputs[i] < 0 else inputs[i]
    return inputs

def sigmoid(inputs):
    for i in range(len(inputs)):
        inputs[i] = 1/(1 + np.exp(-inputs))
    return inputs

class DenseLayer(Layer):
    def __init__(self, units, activation):
        self.units = units
        self.weights = []
        self.bias = np.zeros(units)
        self.activation = activation
        self.outputs = []
        
    def build(self, inputs):
        self.weights = np.random.rand(len(inputs), self.units)
        
    def forward(self, inputs):
        outputs = np.dot(inputs, self.weights) + self.bias
        
        if (self.activation == 'sigmoid'):
            outputs = sigmoid(outputs)
        else:
            outputs = relu(outputs)
        
        return outputs
    
# masukan = [1,2,3,4,5]
# model = DenseLayer(units=3, activation='relu')
# model.build(masukan)
# print(model.forward(masukan))


        
        