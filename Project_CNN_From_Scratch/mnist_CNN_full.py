import numpy as np #just for load data and preprocess data - it is allowed as your words.
from keras.datasets import mnist
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical #just for categorize dataset

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activation_functions import Sigmoid
from activation_functions import Softmax
from activation_functions import ReLU
from losses import binary_cross_entropy, binary_cross_entropy_derivative, binary_cross_entropy_fixed, binary_cross_entropy_derivative_fixed
from train_predict import train, predict
from Pooling import MaxPoolLayer

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 20)

network2 = [
    Convolutional((1, 28, 28), 5, 5,mode="valid"),
    MaxPoolLayer(pool_dimensions=(2, 2), step_size=2),
    Sigmoid(),
    Reshape((5, 12, 12), (5 * 12 * 12, 1)),
    Dense(5 * 12 * 12, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
    #This network worked well
]
network1 = [
    Convolutional((1, 28, 28), 5, 5,mode="valid"),
    MaxPoolLayer(pool_dimensions=(2, 2), step_size=2),
    Sigmoid(),
    Reshape((5, 12, 12), (5 * 12 * 12, 1)),
    Dense(5 * 12 * 12, 100),
    Sigmoid(),
    Dense(100, 2),
    Softmax()
]

# train
train(
    network1,
    binary_cross_entropy_fixed,
    binary_cross_entropy_derivative_fixed,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.01
)
for x, y in zip(x_test, y_test):
    output = predict(network1, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")