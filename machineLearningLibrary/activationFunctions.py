import numpy as np


class ActivationFunction():

    def get(self, z):
        raise NotImplementedError()

    def getDerivative(self, z):
        raise NotImplementedError()


class Sigmoid(ActivationFunction):
    def get(self, z):
        return 1 / (1 + np.exp(-z))


class Tanh(ActivationFunction):
    def get(self, z):
        return np.tanh(z)


class Relu(ActivationFunction):
    def get(self, z):
        return max(0, z)


class LeakyRelu(ActivationFunction):
    def __init__(self, a):
        self.a = a

    def get(self, z):
        return max(self.a * z, z)
