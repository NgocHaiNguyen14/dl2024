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
#Test functions
"""
import math

def test_mean_squared_error():
    # Sample input data
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.1, 2.9, 3.9, 5.2]

    # Calculate expected result
    expected_result = sum((pred - true) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

    # Calculate actual result
    actual_result = mean_squared_error(y_true, y_pred)

    # Print expected and actual results
    print("Expected Mean Squared Error:", expected_result)
    print("Actual Mean Squared Error:", actual_result)

def test_mean_squared_error_prime():
    # Sample input data
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.1, 2.9, 3.9, 5.2]

    # Calculate expected result
    expected_result = [2 * (pred - true) / len(y_true) for true, pred in zip(y_true, y_pred)]

    # Calculate actual result
    actual_result = mean_squared_error_prime(y_true, y_pred)

    # Print expected and actual results
    print("Expected Mean Squared Error Prime:", expected_result)
    print("Actual Mean Squared Error Prime:", actual_result)

def test_binary_cross_entropy():
    # Sample input data
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.8, 0.2, 0.7]

    # Calculate expected result
    expected_result = sum(-true * math.log(pred) - (1 - true) * math.log(1 - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)

    # Calculate actual result
    actual_result = binary_cross_entropy(y_true, y_pred)

    # Print expected and actual results
    print("Expected Binary Cross Entropy:", expected_result)
    print("Actual Binary Cross Entropy:", actual_result)

def test_binary_cross_entropy_prime():
    # Sample input data
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.8, 0.2, 0.7]

    # Calculate expected result
    expected_result = [((1 - true) / (1 - pred) - true / pred) / len(y_true) for true, pred in zip(y_true, y_pred)]

    # Calculate actual result
    actual_result = binary_cross_entropy_prime(y_true, y_pred)

    # Print expected and actual results
    print("Expected Binary Cross Entropy Prime:", expected_result)
    print("Actual Binary Cross Entropy Prime:", actual_result)

# Run tests
test_mean_squared_error()
test_mean_squared_error_prime()
test_binary_cross_entropy()
test_binary_cross_entropy_prime()
"""
