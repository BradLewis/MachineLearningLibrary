import numpy as np


def sigmoid(z):
    """
    Function to return the sigmoid of the provided value
    """
    return 1 / (1 + np.exp(-z))


def relu(z):
    return max(0, z)


def leakyRelu(z, a):
    return max(a * z, z)
