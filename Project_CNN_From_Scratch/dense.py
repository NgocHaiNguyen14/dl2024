import numpy.random as nprandom
from layers import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = nprandom.randn(output_size, input_size)
        self.bias = nprandom.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        output = []
        for i in range(len(self.weights)):
            output.append(sum(self.weights[i][j] * input[j] for j in range(len(input))) + self.bias[i][0])
        return output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = [[0 for _ in range(len(self.weights[0]))] for _ in range(len(self.weights))]
        for i in range(len(output_gradient)):
            for j in range(len(self.input)):
                weights_gradient[i][j] = output_gradient[i] * self.input[j]
        input_gradient = []
        for j in range(len(self.input)):
            input_gradient.append(sum(self.weights[i][j] * output_gradient[i] for i in range(len(output_gradient))))
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] -= learning_rate * weights_gradient[i][j]
        for i in range(len(self.bias)):
            self.bias[i][0] -= learning_rate * output_gradient[i]
        return input_gradient
