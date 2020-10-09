class Layer:

    def forward(self, input):
        pass

    def backward(self, last_gradient, lrate, momentum_rate):
        pass

    def update_weight(self,lrate):
        pass

    def reset_delta(self):
        pass