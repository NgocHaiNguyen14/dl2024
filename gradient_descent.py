def f(x):
  y = x*x;
  return y;

def d(x):
  dx = 2*x;
  return dx;

def gradient_descent(init_input, lr, thresh_hold, n_iteration):
  iteration = 0
  x = init_input

  while iteration < n_iterations:
      x = x - lr * d(x)
      if abs(d(x)) < threshold:
          break

      iteration += 1
  print('With learning rate = ', lr, ':')
  print('Finish with dx = ', d(x))
  print('x = ', x)
  print('Number of iterations: ',iteration)
  print('\n')
if __name__=='__main__':
  init_in = 7
  learning_rate = [0.001,0.005,0.01,0.05,0.1,0.55,2,7]
  threshold = 0.01
  n_iterations = 150
  for lr in learning_rate:
    gradient_descent(init_in,lr,threshold,n_iterations)




