from layers import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return self.elementwise_multiply(output_gradient, self.activation_prime(self.input))

    @staticmethod
    def elementwise_multiply(list1, list2):
        return [[list1[i][j] * list2[i][j] for j in range(len(list1[0]))] for i in range(len(list1))]
