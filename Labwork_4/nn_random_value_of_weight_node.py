import random
import math

class NeuralNetwork:
    def __init__(self, file):
	    with open(file, 'r') as f:
	        lines = f.readlines()
	        layer_sizes = [int(line.strip()) for line in lines if line.strip()]
	        self.num_layers = layer_sizes[0]
	        self.layer_sizes = layer_sizes[1:]

	        # Correctly assign the number of layers and neurons
	        self.num_layers = len(self.layer_sizes) + 1
	        self.layer_sizes = [2] + self.layer_sizes + [2]  # Assuming 2 output neurons

	        # Initialize biases and weights
	        self.biases = [self.init_biases(size) for size in self.layer_sizes[1:]]
	        self.weights = [self.init_weights(self.layer_sizes[i+1], self.layer_sizes[i]) for i in range(self.num_layers - 1)]



    def feedforward(self, a):
        for layer in range(self.num_layers - 1):
            layer_output = []

            for neuron in range(len(self.biases[layer])):
                weighted_sum = 0.0

                for input_index in range(len(a)):
                    weighted_sum += self.weights[layer][neuron][input_index] * a[input_index]

                neuron_output = 1 / (1 + math.exp(-(weighted_sum + self.biases[layer][neuron])))
                layer_output.append(neuron_output)

            a = layer_output

        return a

    # def load_file(self, file):
    #     with open(weights_file, 'r') as file:
    #         lines = file.readlines()
    #         layer_sizes = [int(line.strip()) for line in lines]
    #         self.num_layers = layer_sizes[0]
    #         self.layer_sizes = layer_sizes[1:]

    def init_biases(self, size):
        biases = []
        i = 0
        while i < size:
            biases.append(random.uniform(0, 1))
            i += 1
        return biases

    def init_weights(self, rows, cols):
        weights = []
        i = 0
        while i < rows:
            row = []
            j = 0
            while j < cols:
                row.append(random.uniform(0, 1))
                j += 1
            weights.append(row)
            i += 1
        return weights

if __name__ == '__main__':
    nn = NeuralNetwork("random_value_nn.txt")  # Provide the file name as a string
    print("weights:")
    print(nn.weights)
    print("biases:")
    print(nn.biases)
    
    input_data = [3, 4]
    output = nn.feedforward(input_data)
    print("Output after feedforward:")
    print(output)
