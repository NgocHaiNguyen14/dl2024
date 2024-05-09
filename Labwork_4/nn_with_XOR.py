import random
import math

class NeuralNetwork:
    def __init__(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            layer_sizes = [int(line.strip()) for line in lines if line.strip()]
            self.num_layers = layer_sizes[0]
            self.layer_sizes = layer_sizes[1:]
            self.biases = [self.init_biases(size) for size in self.layer_sizes]
            self.weights = [self.init_weights(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers - 1)]

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

    def init_biases(self, size):
        biases = [random.uniform(0, 1) for _ in range(size)]
        return biases

    def init_weights(self, rows, cols):
        weights = []
        for _ in range(rows):
            row = [random.uniform(0, 1) for _ in range(cols)]
            weights.append(row)
        return weights

if __name__ == '__main__':
    nn = NeuralNetwork("lab4.txt")  # Provide the file name as a string
    print("weights:")
    print(nn.weights)
    print("biases:")
    print(nn.biases)
    
    input_data = [1, 1]
    output = nn.feedforward(input_data)
    print("Output after feedforward:")
    print(output)
