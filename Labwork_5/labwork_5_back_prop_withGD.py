import random
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuron:
    def __init__(self, activation,id):
        self.output = 0
        self.activation = activation
        self.id = id
    def update(self, input):
        self.output = self.activation(input)

    def forward(self):
        return self.output

class BiasNeuron(Neuron):
    def __init__(self, id):
        super().__init__(lambda: 0, id)  # pass id to parent class
        self.output = 1
        self.error = 0

    def update(self, input):
        pass

class Link:
    def __init__(self, fromNeuron, toNeuron, weight):
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron
        self.weight = weight

class Layer:
    def __init__(self, activation, neuron_no, layer_id):
        self.activation = activation
        self.neurons = self._init_neurons(neuron_no, layer_id)

    def _init_neurons(self, neuron_no, layer_id):
        neurons = [BiasNeuron(f"{layer_id}0")]
        for i in range(neuron_no):
            neurons.append(Neuron(self.activation, f"{layer_id}{i+1}"))
        return neurons

    def output(self):
        pass

class LayerLink:
    def __init__(self, fromLayer, toLayer):
        self.fromLayer = fromLayer
        self.toLayer = toLayer
        self.links = self._init_links()

    def _init_links(self):
        links = []
        for fNeuron in self.fromLayer.neurons:
            for tNeuron in self.toLayer.neurons:
                if (isinstance(tNeuron, BiasNeuron)):
                    pass
                links.append(Link(fNeuron, tNeuron, random.uniform(-1,1)))
        return links

class NeuronNetwork:
    def __init__(self, activation, learning_rate, file):
        layer_no, neuron_no = self._get_data(file)
        self.activation = activation
        self.learning_rate = learning_rate
        self.layers = self._init_layers(layer_no, neuron_no)
        self.layerLinks = self._init_layer_links()
        self.errors = []

    def _get_data(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        layer_no = int(lines[0].strip())
        neuron_no = [int(line.strip()) for line in lines[1:]]
        return layer_no, neuron_no


    def _init_layers(self, layer_no, neuron_no):
        layers = []
        for i in range(layer_no):
            layers.append(Layer(self.activation, neuron_no[i], i+1))
        return layers

    def _init_layer_links(self):
        layerLinks = []
        for i in range(len(self.layers)-1):
            layerLinks.append(LayerLink(self.layers[i], self.layers[i+1]))
        return layerLinks

    def predict(self, inputs):
        input_layer = self.layers[0]
        i = 0
        for neuron in input_layer.neurons[1:]:
            neuron.output = inputs[i]
            i += 1

        i = 0  # reset i for layerLinks
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron_input = 0
                for link in self.layerLinks[i].links:
                    if link.toNeuron == neuron:
                        neuron_input += link.weight * link.fromNeuron.output

                neuron.update(neuron_input)
            i += 1  # increment i at the end of each layer

    def _backpropagate(self, targets):
        # Define your threshold and maximum iterations
        threshold = 0.01  # adjust this to your desired threshold
        max_iterations = 50  # adjust this to your desired maximum iterations

        for iteration in range(max_iterations):
            # Calculate errors for output layer
            output_layer = self.layers[-1]
            i = 0
            for neuron in output_layer.neurons[1:]:
                error = targets[i] - neuron.output
                neuron.error = error * (neuron.output * (1 - neuron.output))
                i += 1

            # Backpropagate errors
            for i in range(len(self.layers) - 2, -1, -1):
                j = 0
                for neuron in self.layers[i].neurons[1:]:
                    if isinstance(neuron, BiasNeuron):
                        continue
                    error = sum(link.weight * link.toNeuron.error for link in self.layerLinks[i].links if link.fromNeuron == neuron)
                    neuron.error = error * (neuron.output * (1 - neuron.output))
                    j += 1
                    self.errors.append(error)

            # Update weights
            for i in range(len(self.layers) - 1):
                for link in self.layerLinks[i].links:
                    if not isinstance(link.toNeuron, BiasNeuron):
                        link.weight += self.learning_rate * link.toNeuron.error * link.fromNeuron.output

            # Check if error is below threshold
            if max(self.errors) < threshold:
                break



    def print_weights(self):
        for i, layerLink in enumerate(self.layerLinks):
            print(f"LayerLink {i}:")
            for link in layerLink.links:
                print(f"    Weight from neuron {link.fromNeuron.id} to neuron {link.toNeuron.id}: {link.weight}")
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for input, target in zip(inputs, targets):
                self.predict(input)
                self._backpropagate(target)
            print(f"After epoch {epoch + 1}:")
            self.print_weights()

    def output(self):
        for i in range(len(self.layers)):
            for neuron in self.layers[i].neurons:
                print(f"{i} layer: {neuron.output}")

    def plot_error(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error over Epochs')
        plt.show()

if __name__ == "__main__":
    learning_rate = 0.25

    nn = NeuronNetwork(sigmoid, learning_rate,"nn.txt")
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    nn.train(inputs, targets, epochs=1000)
    # Test the neural network with specific inputs
    test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    print("Testing the neural network:")
    nn.plot_error()
    for input_data in test_inputs:
        nn.predict(input_data)
        print(f"Input: {input_data}, Output: {nn.layers[-1].neurons[-1].output}")
