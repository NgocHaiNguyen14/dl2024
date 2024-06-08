from layers import Layer
from LA import Helper

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return Helper.apply_function(self.activation, self.input)

    def backward(self, output_gradient, learning_rate):
        return Helper.elementwise_multiply(output_gradient, Helper.apply_function(self.activation_prime,self.input))

