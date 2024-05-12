import random
import math

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
                links.append(Link(fNeuron, tNeuron, random.uniform(-1,0.5)))
        return links

class NeuronNetwork:
    def __init__(self, activation, learning_rate, file):
        layer_no, neuron_no = self._get_data(file)
        self.activation = activation
        self.learning_rate = learning_rate
        self.layers = self._init_layers(layer_no, neuron_no)
        self.layerLinks = self._init_layer_links()

    def _get_data(self, file):
        return 3, [2,2,1]

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

        # Update weights
        for i in range(len(self.layers) - 1):
            for link in self.layerLinks[i].links:
                if not isinstance(link.toNeuron, BiasNeuron):
                    link.weight += self.learning_rate * link.toNeuron.error * link.fromNeuron.output


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


if __name__ == "__main__":
    learning_rate = 0.002

    nn = NeuronNetwork(sigmoid, learning_rate,"nn_XOR.txt")
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    nn.train(inputs, targets, epochs=1000)
    nn.output()