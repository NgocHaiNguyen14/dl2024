import math
from layers import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return math.tanh(x)

        def tanh_prime(x):
            return 1 - math.tanh(x) ** 2 

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
            
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return max(0, x)
            
        def relu_prime(x):
            return 1 if x > 0 else 0

        super().__init__(relu, relu_prime)

class Softmax(Layer):
    def forward(self, input):
        exp_input = [math.exp(i[0]) for i in input]
        sum_exp_input = sum(exp_input)
        self.output = [i / sum_exp_input for i in exp_input]
        #print(self.output)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = len(self.output)
        identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        
        jacobian_matrix = [
            [self.output[i] * (identity_matrix[i][j] - self.output[j]) for j in range(n)]
            for i in range(n)
        ]
        
        input_gradient = [sum(jacobian_matrix[i][j] * output_gradient[j] for j in range(n)) for i in range(n)]
        
        return input_gradient
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
