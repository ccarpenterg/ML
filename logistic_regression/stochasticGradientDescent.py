import numpy as np

def costFunction(theta, x, y):
    return -(y * np.log(sigmoid(np.dot(theta.transpose(), x))) + (1 - y) * np.log(sigmoid(1 - np.dot(theta.transpose(), x))))

def grad(theta, x, y, Lambda):
    return (sigmoid(np.dot(theta.transpose(), x)) - y) * x + Lambda * np.r_[0, theta[1:]]

def stochasticGradientDescent(X, labels, iterations, alpha, Lambda):
    m, n = X.shape
    theta = np.zeros(n)
    training_indexes = X.index.values
    np.random.shuffle(training_indexes)
    costsList = []
    for iteration in range(iterations):
        for i in training_indexes:
            alpha = 4 / (1.0 + iteration + i) + 0.01
            x = np.array(X.loc[i])
            y = labels.loc[i]
            costsList.append(costFunction(theta, x, y))
            theta = theta - alpha * grad(theta, x, y, Lambda)
    return theta, costsList

def predict(X, theta):
    X = np.matrix(X)
    prediction = sigmoid(np.dot(X, theta)) >= 0.5
    return prediction.astype(int)

def accuracy(y, prediction):
    accur = (np.array(y) == prediction)
    accur = accur.astype(int)
    return (float(accur.sum()) / accur.size) * 100
