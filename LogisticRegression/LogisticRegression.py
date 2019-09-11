import numpy as np


def train(w, b, X, Y, numIterations, learningRate):
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
    m = X.shape[1]

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    return grads, cost


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_likelihood(features, weights, predictor):
    # Calculates the log-likelihood
    scores = np.dot(features, weights)
    ll = np.sum(predictor*scores - np.log(1 + np.exp(scores)))
    return ll


if __name__ == '__main__':
    pass
