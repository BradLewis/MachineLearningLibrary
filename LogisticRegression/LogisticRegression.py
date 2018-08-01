import numpy as np

def logistic_regression(features, preditor, steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])
    
    for step in range(steps):
        scores = np.dot(features, weights)
        predictions = logistic_function(scores)

        output_error_signal = preditor - predictions
        gradient = np.dot(features.T, output_error_signal) 
        weights += learning_rate * gradient

    return weights

def logistic_function(scores):
    # Returns the sigmoid funciton for the given value
    return 1/ (1 + np.exp(-scores))

def log_likelihood(features, weights, predictor):
    # Calculates the log-likelihood
    scores = np.dot(features, weights)
    normalization_factor = np.exp(scores) + np.exp(-scores)
    ll = (1 / normalization_factor) * np.sum(np.exp(predictor * scores))
    return ll

if __name__ == '__main__':
    pass