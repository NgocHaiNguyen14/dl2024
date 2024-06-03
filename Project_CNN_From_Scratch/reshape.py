from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        reshaped_output = []
        output_index = 0
        for out_dim in self.output_shape:
            dim_size = 1
            for dim in out_dim:
                dim_size *= dim
            reshaped_output.append(input[output_index:output_index+dim_size])
            output_index += dim_size
        return reshaped_output

    def backward(self, output_gradient, learning_rate):
        flattened_gradient = []
        for grad in output_gradient:
            flattened_gradient.extend(grad)
        return flattened_gradient
