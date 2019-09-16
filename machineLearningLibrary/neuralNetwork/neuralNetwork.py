import numpy as np

from machineLearningLibrary.costFunctions import LogCost


def initParams(layerDims):
    """
    Initializes the parameters for the NN.

    Params:
        layerDims - a list of the dimensions for the layers of
            the neural network.

    Returns:
        an initialized dictionary of the weights and biases for
            each layer.
    """
    params = dict()
    L = len(layerDims)

    for l in range(1, L):
        params[f"W{l}"] = np.random.randn(layerDims[l], layerDims[l-1]) * 0.01
        params[f"b{l}"] = np.zeros((layerDims[l], 1))
    return params


def linearForward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linearForwardActivation(A_prev, W, b, activation):
    Z, linearCache = linearForward(A_prev, W, b)
    A = activation.get(Z)
    cache = (linearCache, Z)
    return A, cache


def modelForward(X, params, activations):
    caches = []
    A = X
    L = params // 2

    for l in range(1, L+1):
        A_prev = A
        A, cache = linearForwardActivation(A_prev,
                                           params[f"W{l}"],
                                           params[f"b{l}"],
                                           activations[l])
        caches.append(cache)

    return A, caches


def computeCost(AL, Y):
    return float(np.squeeze(LogCost.get(AL, Y)))


def linearBackward(dZ, cache):
    A_prev, W, _ = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linearBackwardActivation(dA, cache, activation):
    linearCache, Z = cache

    dZ = np.multiply(dA, activation.getDerivative(Z))
    dA_prev, dW, db = linearBackward(dZ, linearCache)

    return dA_prev, dW, db


def modelBackward(AL, Y, caches, activations):
    grads = dict()
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = LogCost.getDerivative(AL, Y)

    currentCache = caches[L-1]
    (grads[f"dA{L-1}"],
     grads[f"dW{L}"],
     grads[f"db{L}"]) = linearBackwardActivation(
        dAL, currentCache, activations[L])

    for l in reversed(range(L-1)):
        currentCache = caches[l]
        (dA_prevTemp,
         dW_temp,
         db_temp) = linearBackwardActivation(
            grads[f"dA{l+1}"], currentCache, activations[l])

        grads[f"dA{l}"] = dA_prevTemp
        grads[f"dW{l+1}"] = dW_temp
        grads[f"db{l+1}"] = db_temp

    return grads


def updateParams(params, grads, learningRate):
    L = len(params) // 2

    for l in range(L):
        params[f"W{l+1}"] = (params[f"W{l+1}"] -
                             learningRate * grads[f"dW{l+1}"])
        params[f"b{l+1}"] = (params[f"b{l+1}"] -
                             learningRate * grads[f"db{l+1}"])
    return params


def train(X, Y, layerDims, activations,
          learningRate=0.01, numIterations=1000, printCost=False):
    costs = []
    params = initParams(layerDims)

    for i in range(0, numIterations):
        AL, caches = modelForward(X, params, activations)
        cost = computeCost(AL, Y)
        grads = modelBackward(AL, Y, caches, activations)
        params = updateParams(params, grads, learningRate)

        if printCost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if printCost and i % 100 == 0:
            costs.append(cost)

    return params


def predict(X, params, activations):
    AL, _ = modelForward(X, params, activations)
    return AL
