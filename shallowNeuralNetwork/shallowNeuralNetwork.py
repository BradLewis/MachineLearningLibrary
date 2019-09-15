import numpy as np

from machineLearningLibrary.activationFunctions import sigmoid


def cost(A2, Y, inParams):
    m = Y.shape
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = - np.sum(logprobs) / m

    return float(np.squeeze(cost))


def forwardPropogation(X, inParams):
    W1 = inParams.get("W1")
    b1 = inParams.get["b1"]
    W2 = inParams.get["W2"]
    b2 = inParams.get["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    outParams = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, outParams
