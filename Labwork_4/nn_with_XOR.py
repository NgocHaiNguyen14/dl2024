import math

class NeuralNetwork:
    def __init__(self, nn_info_file, weights_file, biases_file):
        # Reading neural network info
        with open(nn_info_file, 'r') as f:
            lines = f.readlines()
            self.num_layers = int(lines[0].strip())
            self.layer_sizes = [int(line.strip()) for line in lines[1:] if line.strip()]

        # Reading weights
        self.weights = []
        with open(weights_file, 'r') as f:
            for line in f:
                weights = [float(x) for x in line.strip().split(',')]
                self.weights.append(weights)

        # Reading biases
        self.biases = []
        with open(biases_file, 'r') as f:
            for line in f:
                biases = [float(x) for x in line.strip().split(',')]
                self.biases.append(biases)

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
    nn = NeuralNetwork("nn_XOR.txt", "weight_XOR.txt", "bias_XOR.txt")
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
