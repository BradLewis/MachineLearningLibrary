import numpy as np


class ActivationFunction():
    def get(self, z):
        raise NotImplementedError()

    def getDerivative(self, z):
        raise NotImplementedError()


class Sigmoid(ActivationFunction):
    def get(self, z):
        return 1 / (1 + np.exp(-z))

    def getDerivative(self, z):
        a = self.get(z)
        return a * (1 - a)


class Tanh(ActivationFunction):
    def get(self, z):
        return np.tanh(z)

    def getDerivative(self, z):
        a = self.get(z)
        return 1 - np.power(a, 2)


class Relu(ActivationFunction):
    def get(self, z):
        return max(0, z)

    def getDerivative(self, z):
        if z < 0:
            return 0
        else:
            return 1


class LeakyRelu(ActivationFunction):
    def __init__(self, a):
        self.a = a

    def get(self, z):
        return max(self.a * z, z)

    def getDerivative(self, z):
        if z < 0:
            return self.a
        else:
            return 1
