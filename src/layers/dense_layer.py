import copy

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

        self.delta_w = 0
        self.delta_b = 0
        
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

    def backward(self, last_layer, lrate, momentum_rate):
        dw_output = np.zeros((len(last_layer)))
        if (self.activation == 'sigmoid'):
            dw_output = self.outputs * (1 - self.outputs)
        else:
            for index, element in enumerate(last_layer):
                dw_output[index] *= 1 if self.outputs[index] >= 0 else 0

        gradient = np.dot(-(last_layer - self.outputs), dw_output)

        self.delta_w = momentum_rate * self.delta_w + np.dot(self.outputs, gradient)
        self.bias = momentum_rate * self.delta_b + np.dot(self.outputs, gradient)

        self.weights = self.delta_w - lrate * self.delta_w
        self.bias = self.bias - lrate * self.delta_b

        return gradient
    

        
        