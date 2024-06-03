def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    error = 0
    for i in range(n):
        error += (y_pred[i] - y_true[i]) ** 2
    return error / n

def mean_squared_error_prime(y_true, y_pred):
    n = len(y_true)
    gradient = []
    for i in range(n):
        gradient.append(2 * (y_pred[i] - y_true[i]) / n)
    return gradient

def binary_cross_entropy(y_true, y_pred):
    n = len(y_true)
    error = 0
    for i in range(n):
        error += -y_true[i] * math.log(y_pred[i]) - (1 - y_true[i]) * math.log(1 - y_pred[i])
    return error / n

def binary_cross_entropy_prime(y_true, y_pred):
    n = len(y_true)
    gradient = []
    for i in range(n):
        gradient.append(((1 - y_true[i]) / (1 - y_pred[i]) - y_true[i] / y_pred[i]) / n)
    return gradient
