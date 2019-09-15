import numpy as np


class LogCost():
    @staticmethod
    def get(A2, Y):
        m = Y.shape
        logProb = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
        return - np.sum(logProb) / m

    @staticmethod
    def getDerivative(A2, Y):
        return - (Y / A2) + (1 - Y) / (1 - A2)
