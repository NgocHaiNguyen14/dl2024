import random
import math
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = [[[[random.random() for _ in range(kernel_size)] for _ in range(kernel_size)] for _ in range(input_depth)] for _ in range(depth)]
        self.biases = [[[random.random() for _ in range(input_width - kernel_size + 1)] for _ in range(input_height - kernel_size + 1)] for _ in range(depth)]

    def forward(self, input):
        self.input = input
        self.output = [[[self.biases[i][h][w] for w in range(len(self.biases[i][h]))] for h in range(len(self.biases[i]))] for i in range(self.depth)]
        for i in range(self.depth):
            for j in range(self.input_depth):
                for h in range(len(self.input[j]) - len(self.kernels[i][j]) + 1):
                    for w in range(len(self.input[j][0]) - len(self.kernels[i][j][0]) + 1):
                        conv_result = 0
                        for kh in range(len(self.kernels[i][j])):
                            for kw in range(len(self.kernels[i][j][0])):
                                conv_result += self.input[j][h+kh][w+kw] * self.kernels[i][j][kh][kw]
                        self.output[i][h][w] += conv_result
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = [[[[0 for _ in range(len(self.kernels[i][j][0]))] for _ in range(len(self.kernels[i][j]))] for j in range(self.input_depth)] for i in range(self.depth)]
        input_gradient = [[[0 for _ in range(len(self.input[0][0]))] for _ in range(len(self.input[0]))] for _ in range(self.input_depth)]

        for i in range(self.depth):
            for j in range(self.input_depth):
                for h in range(len(self.input[j]) - len(self.kernels[i][j]) + 1):
                    for w in range(len(self.input[j][0]) - len(self.kernels[i][j][0]) + 1):
                        for kh in range(len(self.kernels[i][j])):
                            for kw in range(len(self.kernels[i][j][0])):
                                kernels_gradient[i][j][kh][kw] += self.input[j][h+kh][w+kw] * output_gradient[i][h][w]
                                input_gradient[j][h+kh][w+kw] += self.kernels[i][j][kh][kw] * output_gradient[i][h][w]

        self.kernels = [[[self.kernels[i][j][kh][kw] - learning_rate * kernels_gradient[i][j][kh][kw] for kw in range(len(self.kernels[i][j][0]))] for kh in range(len(self.kernels[i][j]))] for j in range(self.input_depth)] for i in range(self.depth)]
        self.biases = [[self.biases[i][h][w] - learning_rate * output_gradient[i][h][w] for w in range(len(self.biases[i][h]))] for h in range(len(self.biases[i]))] for i in range(self.depth)]
        return input_gradient
