from context import MachineLearningLibrary
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def generate_data(number_data, random_seed):
    np.random.seed(random_seed)
    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], number_data)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], number_data)
    return x1, x2

def plot_data(simulated_separableish_features, simulated_labels):
    plt.figure(figsize=(12,8))
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = simulated_labels, alpha = .4)
    plt.show()

if __name__ == '__main__':
    number_data = 5000
    x1, x2 = generate_data(number_data, 12)
    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.full(number_data, -1),
                                np.ones(number_data)))

    plot_data(simulated_separableish_features, simulated_labels)

    weights = MachineLearningLibrary.logistic_regression(simulated_separableish_features, simulated_labels, 300000, 5e-5, True)
    clf = LogisticRegression(fit_intercept=True, C=1e15)
    clf.fit(simulated_separableish_features, simulated_labels)

    print(weights)
    print(clf.intercept_, clf.coef_)