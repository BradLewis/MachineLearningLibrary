import numpy as np

from machineLearningLibrary.activationFunctions import sigmoid, tanh


def initParams(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return params


def cost(A2, Y, inParams):
    m = Y.shape
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = - np.sum(logprobs) / m

    return float(np.squeeze(cost))


def forwardPropagation(X, inParams):
    W1 = inParams.get("W1")
    b1 = inParams.get["b1"]
    W2 = inParams.get["W2"]
    b2 = inParams.get["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    outParams = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, outParams


def backwardPropagation(inParams, outParams, X, Y):
    m = X.shape

    W2 = inParams.get("W2")
    A1 = outParams.get("A1")
    A2 = outParams.get("A2")

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

    return gradients


def updateParams(inParams, gradients, learningRate=1.2):
    W1 = inParams.get("W1")
    W2 = inParams.get("W2")
    b1 = inParams.get("b1")
    b2 = inParams.get("b2")

    dW1 = gradients.get("dW1")
    dW2 = gradients.get("dW2")
    db1 = gradients.get("db1")
    db2 = gradients.get("db2")

    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1
    W2 = W2 - learningRate * dW2
    b2 = b2 - learningRate * db2

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return params


def runNeuralNetwork(X, Y, n_h, numIterations=10000, print_cost=False):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = initParams(n_x, n_h, n_y)

    for i in range(0, numIterations):
        A2, cache = forwardPropagation(X, parameters)
        cost = cost(A2, Y, parameters)
        grads = backwardPropagation(parameters, cache, X, Y)
        parameters = updateParams(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters
