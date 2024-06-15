def predict(network, input):
    #print("Predict")
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    print("==================== Train Model ===================")
    error_prev = 0
    for e in range(epochs):
        print(f"=> Traning process: Epoch number: {e}")
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)
            #print("Forward done")
            #print(f"Output forward = {output}")
            #print(f"y = {y}")
            error += loss(y, output)
	    # print("Calculated loss")
            grad = loss_prime(y, output)
            # print("Lost prime")
            # print("Backward start")
            # print(f"Gradient: {grad}")
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
            # print("Backward finished")

        error /= len(x_train)
        if error < 0.01 or abs(error_prev - error) < 0.000000001 :
            print(f"Stopped at epoch number {e} because the error had minor changes.")
            break
        print(f"Lost after epoch {e}: {error}")
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
        error_prev = error
        print(f"Epoch number {e} finished \n")
