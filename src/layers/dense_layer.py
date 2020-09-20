import numpy as np

from layers.base_layer import Layer


def relu(inputs):
    for i in range(len(inputs)):
        inputs[i] = 0 if inputs[i] < 0 else inputs[i]
    return inputs

def sigmoid(inputs):
    for i in range(len(inputs)):
        inputs[i] = 1/(1 + np.exp(-inputs[i]))
    return inputs

class DenseLayer(Layer):
    def __init__(self, units, activation):
        self.units = units
        self.weights = []
        self.bias = np.zeros(units, dtype=float)
        self.activation = activation
        self.outputs = []
        

        
    def forward(self, inputs):
        # Init weights

        inputs =inputs.flatten()
        
        if len(self.weights) == 0:
            self.weights = np.random.rand(len(inputs), self.units)
            
        outputs = np.dot(inputs, self.weights) + self.bias

        if (self.activation == 'sigmoid'):
            outputs = sigmoid(outputs)
        else:
            outputs = relu(outputs)
        return outputs
    

        
        