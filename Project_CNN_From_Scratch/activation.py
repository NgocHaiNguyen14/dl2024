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

def main():
    # Create two matrices of the same size with predefined values
    matrix1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    matrix2 = [
        [9, 8],
        [6, 5,
        [3, 2]
    ]

    # Call the elementwise_multiply method
    result = Activation.elementwise_multiply(matrix1, matrix2)

    # Print the result to verify the element-wise multiplication
    print("Element-wise multiplication result:")
    for row in result:
        print(row)

if __name__ == "__main__":
    main()
