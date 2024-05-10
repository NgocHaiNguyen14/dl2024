import math

class NeuralNetwork:
    def __init__(self, nn_info_file, weights_file, biases_file):
        # Reading neural network info
        with open(nn_info_file, 'r') as f:
            lines = f.readlines()
            layer_sizes = [int(line.strip()) for line in lines if line.strip()]
            self.num_layers = layer_sizes[0]
            self.layer_sizes = layer_sizes[1:]

        # Reading weights
        with open(weights_file, 'r') as f:
            weight_lines = f.readlines()
            self.weights = [[[float(x) for x in line.strip().split(',')] for line in weight_lines if line.strip()][0]]

        # Reading biases
        with open(biases_file, 'r') as f:
            bias_lines = f.readlines()
            self.biases = [[float(x) for x in line.strip().split(',')] for line in bias_lines if line.strip()]

    def feedforward(self, a):
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            layer_output = []

            for neuron_weights, bias in zip(layer_weights, layer_biases):
                weighted_sum = sum(weight * input_data for weight, input_data in zip(neuron_weights, a))
                neuron_output = 1 / (1 + math.exp(-(weighted_sum + bias)))
                layer_output.append(neuron_output)

            a = layer_output

        return a

if __name__ == '__main__':
    nn = NeuralNetwork("input_XOR.txt", "weights_XOR.txt", "biases_XOR.txt")
    print("Neural Network Info:")
    print("Number of Layers:", nn.num_layers)
    print("Layer Sizes:", nn.layer_sizes)
    print("weights:")
    print(nn.weights)
    print("biases:")
    print(nn.biases)
    
    input_data = [1, 1]
    output = nn.feedforward(input_data)
    print("Output after feedforward:")
    print(output)
