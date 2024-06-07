import math
from layers import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return [[math.tanh(item) for item in row] for row in x]

        def tanh_prime(x):
            return [[1 - math.tanh(item) ** 2 for item in row] for row in x]

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return [[1 / (1 + math.exp(-item)) for item in row] for row in x]

        def sigmoid_prime(x):
            s = sigmoid(x)
            return [[item * (1 - item) for item in row] for row in s]

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        exp_input = [[math.exp(item) for item in row] for row in input]
        sum_exp_input = [sum(row) for row in exp_input]
        self.output = [[exp_input[i][j] / sum_exp_input[i] for j in range(len(exp_input[0]))] for i in range(len(exp_input))]
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = len(self.output[0])
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        output_transpose = list(map(list, zip(*self.output)))
        subtracted = [[identity[i][j] - output_transpose[i][j] for j in range(n)] for i in range(n)]
        multiplied = [[self.output[i][j] * subtracted[i][j] for j in range(n)] for i in range(n)]
        return [[sum(multiplied[i][k] * output_gradient[k][j] for k in range(n)) for j in range(len(output_gradient[0]))] for i in range(len(multiplied))]
### Test for program 
"""
def main():
  # Create some sample data
  X = np.random.rand(3, 2)  # Input data with shape (3, 2)

  # Test Tanh activation
  tanh_activation = Tanh()
  tanh_output = tanh_activation.forward(X)
  print("Tanh output:", tanh_output)

  # Test Sigmoid activation
  sigmoid_activation = Sigmoid()
  sigmoid_output = sigmoid_activation.forward(X)
  print("Sigmoid output:", sigmoid_output)

  # Test Softmax layer
  softmax_layer = Softmax()
  softmax_output = softmax_layer.forward(X)
  print("Softmax output:", softmax_output)

if __name__ == "__main__":
  main()
  """