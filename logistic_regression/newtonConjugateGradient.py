import numpy as np
from base import sigmoid

def cost(theta, X, y, m):
    H = sigmoid(np.dot(X, theta))
    return (-1/m) * (np.dot(y.transpose(), np.log(H)) + np.dot((1-y).transpose(), np.log(1-H)))

def gradient(theta, X, y, m):
    H = sigmoid(np.dot(X, theta))
    return (1/m) * np.dot(X.transpose(), H - y)

def newtonConjugateGradient(X, labels, classes, opt):
    k = len(classes)
    m, n = X.shape
    all_thetas = np.zeros(shape=(k, n))
    for c in classes:
        y = (labels == c).astype(int)
        theta0 = np.zeros(n)
        theta = scipy.optimize.fmin_cg(cost, theta0, fprime=gradient, args=(X, y, m))
        all_thetas[c] = theta
