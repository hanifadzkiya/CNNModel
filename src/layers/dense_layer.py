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
        self.inputs = None
        self.delta_w = None
        self.delta_b = np.zeros((self.units))
        
    def forward(self, inputs):
        # Init weights
    
        self.inputs =inputs.flatten()
        if len(self.weights) == 0:
            # self.weights = np.random.rand(len(inputs), self.units)
            self.weights=np.random.uniform(low=-1.0,high=1.01,size=(len(inputs), self.units))
            
        self.outputs = np.dot(inputs, self.weights) + self.bias

        if (self.activation == 'sigmoid'):
            self.outputs = sigmoid(self.outputs)
        else:
            self.outputs = relu(self.outputs)
        return self.outputs
    
    def reset_delta(self):
        self.delta_w = np.zeros((len(self.inputs), self.units))
        self.delta_b = np.zeros((self.units))

    def backward(self, last_gradient, lrate, momentum_rate):
        print("------mulai------")
        if (self.delta_w is None) :
            self.reset_delta()
        gradient = np.zeros((len(last_gradient)))
        print(np.zeros((self.outputs.size)))
        if (self.activation == 'sigmoid'):
            filled_one = np.zeros((len(self.outputs)))
            filled_one.fill(1)
            gradient = np.multiply(self.outputs,(filled_one - self.outputs))
        else:
            for index, element in enumerate(last_gradient):
                gradient[index] = element if self.outputs[index] > 0 else 0
        # gradient = np.dot(-(last_layer - self.outputs), dw_output)
        print("delta_w size : " + str(self.delta_w.shape))
        print("inputs size : " + str(self.inputs.shape))
        print("gradient size : " + str(gradient.shape))
        print("gradient",gradient)
        # ini aku mau coba dulu
        bingungnamanya = np.zeros((len(self.inputs), self.units))
        print(self.inputs.shape)
        print(gradient.shape)
        print(last_gradient.shape)
        print(bingungnamanya.shape)

        for i in range(len(self.inputs)):
            for j in range(self.units):
                bingungnamanya[i,j] = self.inputs[i] * gradient[j] * last_gradient[j]

        

        self.delta_w += bingungnamanya 
        # self.delta_w += momentum_rate * self.delta_w +  np.dot(self.inputs, gradient)
        # self.delta_b += momentum_rate * self.delta_b + np.dot(self.inputs, gradient)

        # self.weights = self.delta_w - lrate * self.delta_w
        # # self.bias = self.bias - lrate * self.delta_b
        # self.delta_w -= lrate * self.delta_w
        # self.delta_bias -= lrate * self.delta_b

        next_gradient = np.zeros((len(self.inputs)))

        for i, el in enumerate(next_gradient):
            sum = 0
            for j, el2 in enumerate(self.weights[i]):
                sum += el2 * gradient[j]
            next_gradient[i] = sum
        
        return copy.deepcopy(next_gradient)

    def update_weight(self,lrate):
        self.weights = self.delta_w - lrate * self.delta_w
        self.bias = self.bias - lrate * self.delta_b

        
        