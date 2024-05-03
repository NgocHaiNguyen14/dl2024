from math import exp, log
import matplotlib.pyplot as plt

def f_pred(x1, x2, y, w0, w1, w2):
    exp_val = -w1 * x1 - w2 * x2 - w0
    if exp_val > 0:
        return 1 / (1 + exp(-exp_val))
    else:
        return exp(exp_val) / (1 + exp(exp_val))

def dw0(x1, x2, y, w0, w1, w2):
    derivative_w0 = 0
    for i in range(len(x1)):
        derivative_w0 += -1 * (y[i] / f_pred(x1[i], x2[i], y[i], w0, w1, w2) + (y[i] - 1) / (1 - f_pred(x1[i], x2[i], y[i], w0, w1, w2))) * (-exp(-w1 * x1[i] - w2 * x2[i] - w0) / (1 + exp(-w1 * x1[i] - w2 * x2[i] - w0)))
    return derivative_w0

def dw1(x1, x2, y, w0, w1, w2):
    derivative_w1 = 0
    for i in range(len(x1)):
        derivative_w1 += -1 * (y[i] / f_pred(x1[i], x2[i], y[i], w0, w1, w2) + (y[i] - 1) / (1 - f_pred(x1[i], x2[i], y[i], w0, w1, w2))) * (-exp(-w1 * x1[i] - w2 * x2[i] - w0) / (1 + exp(-w1 * x1[i] - w2 * x2[i] - w0))) * x1[i]
    return derivative_w1

def dw2(x1, x2, y, w0, w1, w2):
    derivative_w2 = 0
    for i in range(len(x2)):
        derivative_w2 += -1 * (y[i] / f_pred(x1[i], x2[i], y[i], w0, w1, w2) + (y[i] - 1) / (1 - f_pred(x1[i], x2[i], y[i], w0, w1, w2))) * (-exp(-w1 * x1[i] - w2 * x2[i] - w0) / (1 + exp(-w1 * x1[i] - w2 * x2[i] - w0))) * x2[i]
    return derivative_w2

def loss(x1, x2, y, w0, w1, w2):
    n = len(x1)
    J = 0
    for i in range(n):
        pred = f_pred(x1[i], x2[i], y[i], w0, w1, w2)
        if pred == 0:
            pred += 1e-15
        elif pred == 1:
            pred -= 1e-15
        J += (y[i] * log(pred)) + ((1 - y[i]) * log(1 - pred))
    return J / n

def gradient_descent(x1, x2, y, w0, w1, w2, n_iteration, learning_rate, threshold):
    loss_values = []
    for i in range(n_iteration):
        d_w0 = dw0(x1, x2, y, w0, w1, w2)
        d_w1 = dw1(x1, x2, y, w0, w1, w2)
        d_w2 = dw2(x1, x2, y, w0, w1, w2)
        w0 = w0 - learning_rate * d_w0
        w1 = w1 - learning_rate * d_w1
        w2 = w2 - learning_rate * d_w2

        loss_val = loss(x1, x2, y, w0, w1, w2)
        loss_values.append(loss_val)

        if abs(d_w0) < threshold or abs(d_w1) < threshold or abs(d_w2) < threshold:
            i_stop = i
            break
        if i == (n_iteration - 1):
            i_stop = n_iteration
    return w0, w1, w2, i_stop, loss_values

w0 = 1
w1 = 2
w2 = 0

x1 = [1, 2, 3, 4, 1, 2, 2, 1, 6, 7, 8, 9, 10, 8, 9, 7]
x2 = [1, 1, 2, 3, 4, 2, 1, 3, 8, 7, 5, 8, 7, 12, 6, 10]
y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

max_iter = 150
learning_rate = [0.0001, 0.0005, 0.005, 0.01]
threshold = 0.001
n_iteration = 50
i_stop_list = []
loss_list = []

for lr in learning_rate:
    w0, w1, w2, i_stop, loss_values = gradient_descent(x1, x2, y, w0, w1, w2, n_iteration, lr, threshold)
    i_stop_list.append(i_stop)
    plt.plot(loss_values, label='Learning Rate: {}'.format(lr))


plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over Iterations for Different Learning Rates')
plt.legend()
plt.show()

# Plotting points and logistic regression line
# import numpy as np

# plt.scatter(x1[:8], x2[:8], c='blue', label='Class 0')
# plt.scatter(x1[8:], x2[8:], c='red', label='Class 1')
# x_values = np.linspace(0, 10, 100)

# for lr in learning_rate:
#     y_values = -(w1 * x_values + w0) / w2
#     plt.plot(x_values, y_values, label=f'Learning Rate: {lr}')

# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Logistic Regression')
# plt.legend()
# plt.show()

