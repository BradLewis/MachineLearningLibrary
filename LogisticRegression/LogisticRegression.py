import numpy as np

def logistic_regression():
    pass

def logistic_function(values):
    # Returns the sigmoid funciton for the given value
    return 1/ (1 + np.exp(-values))

def log_likelihood(features, weights, predictor):
    # Calculates the log-likelihood
    scores = np.dot(features, weights)
    normalization_factor = np.exp(scores) + np.exp(-scores)
    ll = (1 / normalization_factor) * np.sum(np.exp(predictor * scores))
    return ll

if __name__ == '__main__':
    pass