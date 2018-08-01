import numpy as np

def logistic_regression(features, predictor, steps, learning_rate, add_intercept):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])
    
    for step in range(steps):
        scores = np.dot(features, weights)
        predictions = logistic_function(scores)

        output_error_signal = predictor - predictions
        gradient = np.dot(features.T, output_error_signal) 
        weights += learning_rate * gradient

        if step % 10000 == 0:
            print(log_likelihood(features, weights, predictor))

    return weights

def logistic_function(scores):
    # Returns the sigmoid funciton for the given value
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, weights, predictor):
    # Calculates the log-likelihood
    scores = np.dot(features, weights)
    ll = np.sum(predictor*scores - np.log(1 + np.exp(scores)))
    return ll

if __name__ == '__main__':
    pass