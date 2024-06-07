import random
from layers import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth, mode="valid"):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = [[[[random.random() for _ in range(kernel_size)] for _ in range(kernel_size)] for _ in range(input_depth)] for _ in range(depth)]
        if mode == "valid":
            self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        elif mode == "full":
            self.output_shape = (depth, input_height + kernel_size - 1, input_width + kernel_size - 1)
        else:
            raise ValueError("ERROR in mode: full or valid !!!")
        self.biases = [[[random.random() for _ in range(self.output_shape[2])] for _ in range(self.output_shape[1])] for _ in range(depth)]
        self.mode = mode

    def pad_input(self, input, pad_height, pad_width):
        padded_input = []
        for depth_slice in input:
            padded_slice = [[0] * (len(depth_slice[0]) + 2 * pad_width) for _ in range(pad_height)]
            for row in depth_slice:
                padded_row = [0] * pad_width + row + [0] * pad_width
                padded_slice.append(padded_row)
            padded_slice.extend([[0] * (len(depth_slice[0]) + 2 * pad_width) for _ in range(pad_height)])
            padded_input.append(padded_slice)
        return padded_input

    def forward(self, input):
        if self.mode == "full":
            pad_height = self.kernel_size - 1
            pad_width = self.kernel_size - 1
            input = self.pad_input(input, pad_height, pad_width)
        self.input = input
        self.output = [[[self.biases[i][h][w] for w in range(len(self.biases[i][h]))] for h in range(len(self.biases[i]))] for i in range(self.depth)]
        for i in range(self.depth):
            for j in range(self.input_depth):
                for h in range(len(self.input[j]) - len(self.kernels[i][j]) + 1):
                    for w in range(len(self.input[j][0]) - len(self.kernels[i][j][0]) + 1):
                        conv_result = 0
                        for kh in range(len(self.kernels[i][j])):
                            for kw in range(len(self.kernels[i][j][0])):
                                conv_result += self.input[j][h + kh][w + kw] * self.kernels[i][j][kh][kw]
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
                                kernels_gradient[i][j][kh][kw] += self.input[j][h + kh][w + kw] * output_gradient[i][h][w]
                                input_gradient[j][h + kh][w + kw] += self.kernels[i][j][kh][kw] * output_gradient[i][h][w]

        for i in range(self.depth):
            for j in range(self.input_depth):
                for kh in range(len(self.kernels[i][j])):
                    for kw in range(len(self.kernels[i][j][0])):
                        self.kernels[i][j][kh][kw] -= learning_rate * kernels_gradient[i][j][kh][kw]

        for i in range(self.depth):
            for h in range(len(self.biases[i])):
                for w in range(len(self.biases[i][h])):
                    self.biases[i][h][w] -= learning_rate * output_gradient[i][h][w]

        return input_gradient

#Test methods        
"""
def main():
    # Define input shape (depth, height, width)
    input_shape = (3, 5, 5)

    # Define kernel size and depth of the convolutional layer
    kernel_size = 3
    depth = 8

    # Initialize Convolutional layer
    conv_layer = Convolutional(input_shape, kernel_size, depth, mode="valid")

    # Generate random input tensor with the shape (3, 5, 5)
    input_data = [[[random.random() for _ in range(5)] for _ in range(5)] for _ in range(3)]

    # Print input tensor
    print("Input data:")
    for i, layer in enumerate(input_data):
        print("Depth {}:".format(i))
        for row in layer:
            print(row)

    # Perform forward pass
    forward_output = conv_layer.forward(input_data)
    print("\nForward output:")
    for i, layer in enumerate(forward_output):
        print("Depth {}:".format(i))
        for row in layer:
            print(row)

    # Generate random output gradient with the same shape as the output
    output_gradient = [[[random.random() for _ in range(len(forward_output[0][0]))] for _ in range(len(forward_output[0]))] for _ in range(depth)]

    # Perform backward pass with a learning rate of 0.01
    learning_rate = 0.01
    backward_output = conv_layer.backward(output_gradient, learning_rate)
    print("\nBackward output (input gradient):")
    for i, layer in enumerate(backward_output):
        print("Depth {}:".format(i))
        for row in layer:
            print(row)

if __name__ == "__main__":
    main()
"""