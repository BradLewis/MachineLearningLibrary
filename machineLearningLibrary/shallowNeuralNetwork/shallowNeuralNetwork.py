import numpy as np

from machineLearningLibrary.activationFunctions import Sigmoid, Tanh
from machineLearningLibrary.costFunctions import LogCost


def initParams(n_x, n_h, n_y):
    """
    Initialises the parameters of the NN based on the number of input,
        output and hidden layers.

    Params:
        n_x - size of the input layer.
        n_h - size of the hidden layer.
        n_h - size of the output layer.

    Returns:
        params - a dict containing the initialized values for the NN, 
            W1, W2, b1 and b2.
    """
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


def cost(A2, Y):
    """
    Calculates the cost for a given output with the expected out.

    Params:
        A2 - the calculated output.
        Y - the expecte output.

    Returns
        a float value for the cost.
    """
    return float(np.squeeze(LogCost.get(A2, Y)))


def forwardPropagation(X, inParams):
    """
    Calculates the output based on the input data and the params
        of the NN.

    Params:
        X - the input data.
        inParams - the params of the NN.

    Returns:
        A2 - the output of the NN.
        outParams - a dict containing the calculated params A1, A2,
            Z1 and Z2 needed for the backward propagation.
    """
    W1 = inParams.get("W1")
    b1 = inParams.get["b1"]
    W2 = inParams.get["W2"]
    b2 = inParams.get["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = Tanh().get(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Sigmoid().get(Z2)

    outParams = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, outParams


def backwardPropagation(inParams, outParams, X, Y,
                        hiddenLayerAct=None,
                        finalLayerAct=None):
    """
    Performs backward propagation based on the input and output
        of the NN.

    Params:
        inParams - the parameters for the NN, W1, W2, b1, b2.
        outParams - the calculated output parameters for the run
            of the NN, A1, A2, Z1, Z2.
        X - the input data.
        Y - the expected output data.
        hiddenLayerAct - the activation function for the hidden layer. If
            None, it will default to the Tanh function. Defaults None.
        finalLayerAct - the activation function for the final output layer.
            If None, it will default to the Sigmoid function. Defaults None.

    Returns:
        a dict of the gradients for the inParams, dW1, dW2, db1, db2.
    """
    if hiddenLayerAct is None:
        hiddenLayerAct = Tanh()
    if finalLayerAct is None:
        finalLayerAct = Sigmoid()

    m = X.shape

    W2 = inParams.get("W2")
    A1 = outParams.get("A1")
    A2 = outParams.get("A2")

    dZ2 = LogCost.getDerivative(A2, Y) * finalLayerAct.getDerivative(A2)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * hiddenLayerAct.getDerivative(A1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

    return gradients


def updateParams(inParams, gradients, learningRate=1.2):
    """
    Updates the parameters of the NN based on the gradients and
        the learning rate.

    Params:
        inParams - the parameters of the NN.
        gradients - the gradients for the parameters of the NN.
        learningRate - the learning rate for the NN. Defaults 1.2.

    Returns:
        a dict of the updated values for the NN.
    """
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


def train(X, Y, n_h, numIterations=10000, printCost=False):
    """
    Trains a shallow NN.

    Params:
        X - the training input data.
        Y - the expected output for the input data X.
        n_h - the number of nodes in the hidden layer.
        numIterations - number of iterations to train the NN.
            Defaults 10000.
        printCost - whether to print updates for the cost function
            every 1000 iterations. Defaults False.

    Returns:
        a dict of the params W1, W2, b1, b2 for the trained NN.
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]

    params = initParams(n_x, n_h, n_y)

    for i in range(0, numIterations):
        A2, cache = forwardPropagation(X, params)
        cost = cost(A2, Y)
        grads = backwardPropagation(params, cache, X, Y)
        params = updateParams(params, grads)

        if printCost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return params


def predict(X, params):
    """
    Convience method for calculating the output of the NN for a
        given input.

    Params:
        X - the input data.
        params - the params of the NN, W1, W2, b1, b2.

    Returns:
        the predicted value A2 for X.
    """
    A2, _ = forwardPropagation(X, params)
    return A2
