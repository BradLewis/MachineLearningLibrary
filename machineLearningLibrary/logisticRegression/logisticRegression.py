import numpy as np

from machineLearningLibrary.activationFunctions import sigmoid


def predict(w, b, X):
    """
    Params:
        w - the weights for the model
        b - the bias for the model
        X - the inputs to evaluate with the model with size (size, num of inputs)

    Returns:
        predictions - list of the predictions for all the inputs
    """
    w = w.reshape(X.shape[0], 1)
    Z = np.dot(w.T, X) + b
    predictions = sigmoid(Z)
    return predictions


def train(w, b, X, Y, numIterations, learningRate):
    """
    Trains the model.

    Params:
        w - the initial weights
        b - the initial bias
        X - the training input data
        Y - the training output data
        numIterations - number of iterations to train the model with the data
        learningRate - the learning rate for each iteration

    Returns:
        params - dict of the final params for the model
        grads - dict of the final gradients for the model
        costs - list of the costs updated every 100 iterations
    """
    costs = []

    for i in range(numIterations):
        grads, cost = propagate(w, b, X, Y)

        w = w - learningRate * grads['dw']
        b = b - learningRate * grads['db']
        if i % 100 == 0:
            costs.append(cost)

        params = {"w": w, "b": b}
        return params, grads, costs


def propagate(w, b, X, Y):
    """
    Does one forward and backward propogation.

    Params:
        w - the current weights
        b - the current bias
        X - the training input data
        Y - the training output data

    Returns:
        grads - dict of the gradients for the propogation
        cost - the value of the cost function for the propogation
    """
    m = X.shape[1]

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    return grads, cost


def log_likelihood(features, weights, predictor):
    # Calculates the log-likelihood
    scores = np.dot(features, weights)
    ll = np.sum(predictor*scores - np.log(1 + np.exp(scores)))
    return ll


if __name__ == '__main__':
    pass
