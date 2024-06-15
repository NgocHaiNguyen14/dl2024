from layers import Layer

class MaxPoolLayer(Layer):
    def __init__(self, pool_dimensions=(2, 2), step_size=2, logging=False):
        self.pool_h, self.pool_w = pool_dimensions
        self.stride_h, self.stride_w = step_size, step_size
        self.logging = logging
        self.input_data = None
        self.output_data = None
    
    def forward(self, input_matrix):
        super().forward(input_matrix)
        self.input_data = input_matrix
        self.output_data = []
        if isinstance(input_matrix[0][0], list):
            channels = len(input_matrix)
            for channel in input_matrix:
                out_height = (len(channel) - self.pool_h) // self.stride_h + 1
                out_width = (len(channel[0]) - self.pool_w) // self.stride_w + 1
                result = [[0] * out_width for _ in range(out_height)]

                for i in range(out_height):
                    for j in range(out_width):
                        start_i = i * self.stride_h
                        start_j = j * self.stride_w
                        pooling_region = [row[start_j:start_j + self.pool_w] for row in channel[start_i:start_i + self.pool_h]]
                        result[i][j] = max(max(region) for region in pooling_region)
                self.output_data.append(result)
        else:
            out_height = (len(input_matrix) - self.pool_h) // self.stride_h + 1
            out_width = (len(input_matrix[0]) - self.pool_w) // self.stride_w + 1
            result = [[0] * out_width for _ in range(out_height)]

            for i in range(out_height):
                for j in range(out_width):
                    start_i = i * self.stride_h
                    start_j = j * self.stride_w
                    pooling_region = [row[start_j:start_j + self.pool_w] for row in input_matrix[start_i:start_i + self.pool_h]]
                    result[i][j] = max(max(region) for region in pooling_region)
            self.output_data = result

        return self.output_data

    def backward(self, grad_output, learning_rate):
        super().backward(grad_output, learning_rate)
        grad_input = []
        if isinstance(grad_output[0][0], list): 
            for idx in range(len(self.input_data)):
                grad_input.append(self.backward3D(grad_output[idx], idx))
        else:
            grad_input = self.backward2D(grad_output)

        return grad_input
        
    def input_shape(self):
        if isinstance(self.input_data[0][0], list): 
            depth, height, width = len(self.input_data), len(self.input_data[0]), len(self.input_data[0][0])
            return depth, height, width
        else:
            height, width = len(self.input_data), len(self.input_data[0])
            return height, width

    def output_shape(self):
        if isinstance(self.output_data[0][0], list):
            depth, height, width = len(self.output_data), len(self.output_data[0]), len(self.output_data[0][0])
            return depth, height, width
        else:
            height, width = len(self.output_data), len(self.output_data[0])
            return height, width

    def backward3D(self, grad_output, channel_idx):
        grad_input_channel = [[0] * len(self.input_data[0][0]) for _ in range(len(self.input_data[0]))]

        for i in range(len(grad_output)):
            for j in range(len(grad_output[0])):
                start_i = i * self.stride_h
                start_j = j * self.stride_w
                pooling_region = [row[start_j:start_j + self.pool_w] for row in self.input_data[channel_idx][start_i:start_i + self.pool_h]]
                max_val = max(max(region) for region in pooling_region)
                
                for ii in range(self.pool_h):
                    for jj in range(self.pool_w):
                        if pooling_region[ii][jj] == max_val:
                            grad_input_channel[start_i + ii][start_j + jj] += grad_output[i][j]

        return grad_input_channel
    
    def backward2D(self, grad_output):
        grad_input = [[0] * len(self.input_data[0]) for _ in range(len(self.input_data))]

        for i in range(len(grad_output)):
            for j in range(len(grad_output[0])):
                start_i = i * self.stride_h
                start_j = j * self.stride_w
                pooling_region = [row[start_j:start_j + self.pool_w] for row in self.input_data[start_i:start_i + self.pool_h]]
                max_val = max(max(region) for region in pooling_region)
                
                for ii in range(self.pool_h):
                    for jj in range(self.pool_w):
                        if pooling_region[ii][jj] == max_val:
                            grad_input[start_i + ii][start_j + jj] += grad_output[i][j]

        return grad_input
