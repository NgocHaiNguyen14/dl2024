import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

with open('data.csv', 'r') as file:
    next(file)  # Skip the header
    for line in file:
        row = line.strip().split(',')
        x_read = float(row[0])/1000
        y_read = float(row[1])
        x.append(x_read)
        y.append(y_read)

      # i need to scale value of x because the proportion between x and y are too small
      # i want to test with your initialization w1 = 1 and w0 =0, so that proportion x/y should near to 1
x = np.array(x)
y = np.array(y)
#print(x)
#print(y)
def f(x,w0,w1):
  return w1 * x + w0

def MSE(x,y,w0,w1): # define loss function of linear regressioon
  y_pred = f(x,w0,w1) # array of prediction's value
  total_square_error = 0
  for i in range(len(x)):
    total_square_error += ((y[i] - y_pred[i])**2)/2
  mean_square_error = total_square_error/len(x)
  return mean_square_error

def d_w0(x,y,w0,w1): # dL/dw0
  dw0 = 0
  for i in range(len(x)):
    dw0 += w1*x[i] + w0 - y[i]
  return dw0

def d_w1(x,y,w0,w1): #dL/dw1
  dw1 = 0
  for i in range(len(x)):
    dw1 += (w1*x[i] + w0 - y[i]) * x[i]
  return dw1

def gradient_descent_w0_w1(x,y,w0,w1,n_iteration,learning_rate,threshold):
  for i in range(n_iteration):
    dw0 = d_w0(x,y,w0,w1)
    dw1 = d_w1(x,y,w0,w1)
    w0 = w0 - learning_rate*dw0
    w1 = w1 - learning_rate*dw1

    loss = MSE(x,y,w0,w1)
    i_stop = i;
    if abs(dw1) < threshold or abs(dw0)< threshold:
      i_stop = i
      break
    if i == (n_iteration - 1):
      i_stop = n_iteration
  return w0,w1,i_stop,loss

def print_result(w0,w1, i_stop,learning_rate, n_iteration):
  print("With learning rate = ", learning_rate)
  if i_stop < n_iteration:
      print("Stop in iteration number: ", i_stop)
      print("w0 = ", w0, "and", "w1 = ", w1, "\n")
  else:
      print("Stop because reached max iteration = 150")
      print("w0 = ", w0, "and", "w1 = ", w1, "\n")



def plot_data_and_regression_line(x, y, regression_lines, learning_rates):
    # Plot the data points
    plt.scatter(x, y, color='blue', label='Data points')

    # Plot each regression line
    for i, (w0, w1) in enumerate(regression_lines):
        y_pred = f(x, w0, w1)
        label = f'Regression line (LR={learning_rates[i]})'
        plt.plot(x, y_pred, label=label)

    # Set the range of the axes
    plt.xlim(1.6, 2.1)
    plt.ylim(1, 4)

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data points and Regression lines')

    # Add a legend
    plt.legend()

    # Show grid for better visibility
    plt.grid(True)

    # Show the plot
    plt.show()
import matplotlib.pyplot as plt

def plot_iterations_vs_loss(learning_rates, i_stop_list, loss_list):
    # Create a scatter plot with each point representing a learning rate
    plt.figure(figsize=(10, 6))

    # Plotting each learning rate as a point in the graph
    for i, lr in enumerate(learning_rates):
        plt.scatter(i_stop_list[i], loss_list[i], label=f'LR: {lr}', s=100)

    # Labeling the axes and setting the title
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss at Stopping Point')
    plt.title('Iterations vs Loss for Different Learning Rates')

    # Adding a legend
    plt.legend(title='Learning Rates')

    # Display the plot
    plt.show()

def calculate_initial_w0_w1(x,y):
	mean_x = calculate_mean(x)
	mean_y = calculate_mean(y)
	numerator = 0
	denominator = 0
	for x0,y0 in zip(x,y):
	  x_deviation = x0 - mean_x
	  y_deviation = y0 - mean_y

	  numerator += y_deviation*x_deviation
	  denominator += x_deviation**2

	w1_si = numerator/denominator
	w0_si = mean_y - w1_si * mean_x
	return w0_si,w1_si



#### with w0 = 0 and w1 = 1

 
w0 = 0
w1 = 1
max_iter = 150
learning_rate = [0.00005,0.0001,0.0005,0.001,0.003,0.005,0.007,0.009,0.01]
threshold = 0.001

i_stop_list = []
loss_list = []

for lr in learning_rate:
  w0_best, w1_best, i_stop,loss = gradient_descent_w0_w1(x, y, w0, w1, max_iter, lr, threshold)
  i_stop_list.append(i_stop)
  loss_list.append(loss)
# print(i_stop_list)
# print(loss_list)
plot_iterations_vs_loss(learning_rate, i_stop_list, loss_list)
regression_lines = []
for lr in learning_rate:
  w0_best, w1_best, i_stop,loss = gradient_descent_w0_w1(x, y, w0, w1, max_iter, lr, threshold)
  print_result(w0_best, w1_best, i_stop, lr, max_iter)
  regression_lines.append((w0_best, w1_best))

plot_data_and_regression_line(x, y, regression_lines, learning_rate)


#### with calculated w0 and w1
w0, w1 = calculate_initial_w0_w1(x,y)
max_iter = 150
learning_rate = [0.00005,0.0001,0.0005,0.001,0.003,0.005,0.007,0.009,0.01]
threshold = 0.001

i_stop_list = []
loss_list = []

for lr in learning_rate:
  w0_best, w1_best, i_stop,loss = gradient_descent_w0_w1(x, y, w0, w1, max_iter, lr, threshold)
  i_stop_list.append(i_stop)
  loss_list.append(loss)
# print(i_stop_list)
# print(loss_list)
plot_iterations_vs_loss(learning_rate, i_stop_list, loss_list)
regression_lines = []
for lr in learning_rate:
  w0_best, w1_best, i_stop,loss = gradient_descent_w0_w1(x, y, w0, w1, max_iter, lr, threshold)
  print_result(w0_best, w1_best, i_stop, lr, max_iter)
  regression_lines.append((w0_best, w1_best))

plot_data_and_regression_line(x, y, regression_lines, learning_rate)
