import numpy as np


def costFunction(theta, x, y):
    return -(y * np.log(sigmoid(np.dot(theta.transpose(), x))) + (1 - y) * np.log(sigmoid(1 - np.dot(theta.transpose(), x))))

def grad(theta, x, y, Lambda):
    return (sigmoid(np.dot(theta.transpose(), x)) - y) * x + Lambda * np.r_[0, theta[1:]]

def stochasticGradientDescent(df, label, iterations, alpha, Lambda):
    features = list(df.columns)
    features.remove(label)
    theta = np.zeros(len(features))
    training_indexes = df.index.values
    np.random.shuffle(training_indexes)
    costsList = []
    for iteration in range(iterations):
        for i in training_indexes:
            alpha = 4 / (1.0 + iteration + i) + 0.01
            x = np.array(df.loc[i, features[0]:features[-1]])
            y = df.loc[i, label]
            costsList.append(costFunction(theta, x, y))
            theta = theta - alpha * grad(theta, x, y, Lambda)
    return theta, costsList

