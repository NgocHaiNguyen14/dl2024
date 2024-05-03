from math import exp, log
import matplotlib.pyplot as plt

def z(x1, x2,y , w0, w1, w2, w3, w4, w5, w6, w7):
  z_value =  -w1 * (x1**3) - w2 * (x1**2) - w3*x1 - w0 -w5 * (x2**3) - w6 * (x2**2) - w7*x2 - w4
  return z_value

def f_pred(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
    exp_val = z(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
    if exp_val > 0:
        return 1 / (1 + exp(-exp_val))
    else:
        return exp(exp_val) / (1 + exp(exp_val))
def dw0(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
  d_w0 = 0
  for i in range(len(x1)):
    z_val = z(x1[i], x2[i], y[i], w0, w1, w2, w3, w4, w5, w6, w7)
    yi = f_pred(x1[i], x2[i], y[i], w0, w1, w2,w3, w4,w5,w6,w7)
    d_w0 += (y[i]/yi + (y[i] - 1)/(1-yi))*(-exp(-z_val)/(1+exp(-z_val)))
  return d_w0

def dw4(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
  d_w4 = 0
  for i in range(len(x1)):
    z_val = z(x1[i], x2[i], y[i], w0, w1, w2, w3, w4, w5, w6, w7)
    yi = f_pred(x1[i], x2[i], y[i], w0, w1, w2,w3, w4,w5,w6,w7)
    d_w4 += (y[i]/yi + (y[i]-1)/(1-yi))*(-exp(-z_val)/(1+exp(-z_val)))
  return d_w4

def dw1(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
  d_w1 = 0
  for i in range(len(x1)):
    z_val = z(x1[i], x2[i], y[i], w0, w1, w2, w3, w4, w5, w6, w7)
    yi = f_pred(x1[i], x2[i], y[i], w0, w1, w2,w3, w4,w5,w6,w7)
    d_w1 += (y[i]/yi + (y[i]-1)/(1-yi))*(-exp(-z_val)/(1+exp(-z_val)))*(x1[i]**3)
  return d_w1

def dw5(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
  d_w5 = 0
  for i in range(len(x1)):
    z_val = z(x1[i], x2[i], y[i], w0, w1, w2, w3, w4, w5, w6, w7)
    yi = f_pred(x1[i], x2[i], y[i], w0, w1, w2,w3, w4,w5,w6,w7)
    d_w5 += (y[i]/yi + (y[i]-1)/(1-yi))*(-exp(-z_val)/(1+exp(-z_val)))*(x2[i]**3)
  return d_w5

def dw2(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
  d_w2 = 0
  for i in range(len(x1)):
    z_val = z(x1[i], x2[i], y[i], w0, w1, w2, w3, w4, w5, w6, w7)
    yi = f_pred(x1[i], x2[i], y[i], w0, w1, w2,w3, w4,w5,w6,w7)
    d_w2 += (y[i]/yi + (y[i]-1)/(1-yi))*(-exp(-z_val)/(1+exp(-z_val)))*(x1[i]**2)
  return d_w2

def dw6(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
  d_w6 = 0
  for i in range(len(x1)):
    z_val = z(x1[i], x2[i], y[i], w0, w1, w2, w3, w4, w5, w6, w7)
    yi = f_pred(x1[i], x2[i], y[i], w0, w1, w2,w3, w4,w5,w6,w7)
    d_w6 += (y[i]/yi + (y[i]-1)/(1-yi))*(-exp(-z_val)/(1+exp(-z_val)))*(x2[i]**2)
  return d_w6

def dw3(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
  d_w3 = 0
  for i in range(len(x1)):
    z_val = z(x1[i], x2[i], y[i], w0, w1, w2, w3, w4, w5, w6, w7)
    yi = f_pred(x1[i], x2[i], y[i], w0, w1, w2,w3, w4,w5,w6,w7)
    d_w3 += (y[i]/yi + (y[i]-1)/(1-yi))*(-exp(-z_val)/(1+exp(-z_val)))*(x1[i])
  return d_w3

def dw7(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7):
  d_w7 = 0
  for i in range(len(x1)):
    z_val = z(x1[i], x2[i], y[i], w0, w1, w2, w3, w4, w5, w6, w7)
    yi = f_pred(x1[i], x2[i], y[i], w0, w1, w2,w3, w4,w5,w6,w7)
    d_w7 += (y[i]/yi + (y[i]-1)/(1-yi))*(-exp(-z_val)/(1+exp(-z_val)))*(x2[i])
  return d_w7

def loss(x1, x2, y, w0, w1, w2,w3, w4,w5,w6,w7):
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

def gradient_descent(x1, x2, y, w0, w1, w2,w3,w4,w5,w6,w7, n_iteration, learning_rate, threshold):
    loss_values = []
    for i in range(n_iteration):
        d_w0 = dw0(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
        d_w1 = dw1(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
        d_w2 = dw2(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
        d_w3 = dw0(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
        d_w4 = dw1(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
        d_w5 = dw2(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
        d_w6 = dw1(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
        d_w7 = dw2(x1, x2, y, w0, w1, w2, w3, w4, w5, w6, w7)
        w0 = w0 - learning_rate * d_w0
        w1 = w1 - learning_rate * d_w1
        w2 = w2 - learning_rate * d_w2
        w3 = w3 - learning_rate * d_w3
        w4 = w4 - learning_rate * d_w4
        w5 = w5 - learning_rate * d_w5
        w6 = w6 - learning_rate * d_w6
        w7 = w7 - learning_rate * d_w7

        loss_val = loss(x1, x2, y, w0, w1, w2,w3, w4,w5,w6,w7)
        loss_values.append(loss_val)

        if abs(d_w0) < threshold or abs(d_w1) < threshold or abs(d_w2) < threshold:
            i_stop = i
            break
        if i == (n_iteration - 1):
            i_stop = n_iteration
    return w0, w1, w2, w3, w4, w5, w6, w7, i_stop, loss_values

w0 = 1
w1 = 2
w2 = 0
w3 = 1.7
w4 = 3
w5 = 2.5
w6 = 1.2
w7 = 0.5

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
    w0, w1, w2, w3, w4, w5, w6, w7, i_stop, loss_values = gradient_descent(x1, x2, y, w0, w1, w2,w3,w4,w5,w6,w7, n_iteration, learning_rate, threshold)
    i_stop_list.append(i_stop)
    plt.plot(loss_values, label=f'Learning Rate: {lr}')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over Iterations for Different Learning Rates')
plt.legend()
plt.show()

# Plotting points and logistic regression line
import numpy as np

plt.scatter(x1[:8], x2[:8], c='blue', label='Class 0')
plt.scatter(x1[8:], x2[8:], c='red', label='Class 1')
x_values = np.linspace(0, 10, 100)

for lr in learning_rate:
    y_values = -(w1 * x_values + w0) / w2
    plt.plot(x_values, y_values, label=f'Learning Rate: {lr}')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression')
plt.legend()
plt.show()

