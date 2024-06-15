import numpy as np
from keras.datasets import mnist
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activation_functions import Sigmoid
from activation_functions import Tanh
from losses import mean_squared_error, mean_squared_error_prime
from train_predict import train, predict

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    #y = np_utils.to_categorical(y)
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 20)

# Multi Layers Perceptrons
# neural network 1
network1 = [
    #Reshape((1, 28, 28), ( 28 * 28, 1)),
    Dense(28 * 28, 1000),
    Tanh(),
    Dense(1000,100),
    Tanh(),
    Dense(100, 10),
    Tanh()
]

# train network1
train(network1, mean_squared_error, mean_squared_error_prime, x_train, y_train, epochs=100, learning_rate=0.01)

for x, y in zip(x_test, y_test):
    output = predict(network1, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")