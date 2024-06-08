from layers import Layer
from LA import Helper
import random

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = [[random.gauss(0, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.bias = [[random.gauss(0, 1)] for _ in range(output_size)]

    def forward(self, input):
        # print("-------------------------DENSE FORWARD-------------------------")
        self.input = input
        return Helper.add(Helper.dot_product(self.weights, self.input), self.bias)

    def backward(self, output_gradient, learning_rate):
        weights_gradient = Helper.dot_product(output_gradient, Helper.transpose(self.input))
        input_gradient = Helper.dot_product(Helper.transpose(self.weights), output_gradient)

        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] -= learning_rate * weights_gradient[i][j]

        for i in range(len(self.bias)):
            self.bias[i][0] -= learning_rate * output_gradient[i][0]

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
