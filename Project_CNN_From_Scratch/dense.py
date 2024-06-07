import numpy as np #import numpy just for testing
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


#Test methods        
"""
def main():
    # Initialize Dense layer with input size of 3 and output size of 2
    dense_layer = Dense(input_size=5, output_size=2)
    
    # Generate random input data of size 3
    input_data = nprandom.randn(5)
    print("Forward input:", input_data )
    # Perform forward pass
    forward_output = dense_layer.forward(input_data)
    print("Forward output:", forward_output)
    
    # Generate random output gradients of size 2 to simulate backpropagation gradients
    output_gradient = nprandom.randn(2)
    
    # Perform backward pass with a learning rate of 0.01
    learning_rate = 0.01
    backward_output = dense_layer.backward(output_gradient, learning_rate)
    print("Backward output (input gradient):", backward_output)

if __name__ == "__main__":
    main()
"""