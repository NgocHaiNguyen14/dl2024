from layers import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return self.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate): 
        return self.reshape(output_gradient, self.input_shape)

    def flatten(self, array):
        if isinstance(array, int):
            return [array]
        flat = []
        for item in array:
            flat.extend(self.flatten(item))
        return flat

    def reshape(self, array, shape):
        reshaped = []
        flat = self.flatten(array)
        current_index = 0

        def build_dim(shape):
            nonlocal current_index
            if len(shape) == 1:
                sub_array = flat[current_index:current_index + shape[0]]
                current_index += shape[0]
                return sub_array
            return [build_dim(shape[1:]) for _ in range(shape[0])]

        reshaped = build_dim(shape)
        return reshaped


#Test methods
"""
def main():
    # Define input shape and output shape
    input_shape = (2, 3, 4)  # Depth: 2, Height: 3, Width: 4
    output_shape = [(2, 6), (2, 2, 3)]  # Two output shapes: (2, 6) and (2, 2, 3)

    # Create Reshape layer
    reshape_layer = Reshape(input_shape, output_shape)

    # Define input data
    input_data = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
    ]

    # Perform forward pass
    reshaped_output = reshape_layer.forward(input_data)

    # Define expected outputs
    expected_outputs = [
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ]
    ]

    # Print actual and expected results
    for i, (actual_output, expected_output) in enumerate(zip(reshaped_output, expected_outputs), start=1):
        print(f"Output {i}:")
        print("Actual   :", actual_output)
        print("Expected :", expected_output)
        print()

    # Perform backward pass with random output gradients
    output_gradient = [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]],
                       [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]]
    input_gradient = reshape_layer.backward(output_gradient, learning_rate=0.01)

    # Print backward pass result
    print("\nBackward Pass Input Gradient:")
    print(input_gradient)

if __name__ == "__main__":
    main()
"""