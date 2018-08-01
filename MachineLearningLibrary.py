import LogisticRegression.LogisticRegression as lg

def logistic_regression(features, target, steps, learning_rate, add_intercept = False):
    return lg.logistic_regression(features, target, steps, learning_rate, add_intercept)

def logistic_function(scores):
    return lg.logistic_function(scores)

if __name__ == "__main__":
    pass