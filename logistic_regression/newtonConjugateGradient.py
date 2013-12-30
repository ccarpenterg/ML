import numpy as np
from scipy import optimize
from base import sigmoid

def cost(theta, X, y, m):
    H = sigmoid(np.dot(X, theta))
    return (-1./m) * (np.dot(y.transpose(), np.log(H)) + np.dot((1-y).transpose(), np.log(1-H)))

def gradient(theta, X, y, m):
    H = sigmoid(np.dot(X, theta))
    return (1./m) * np.dot(X.transpose(), H - y)

def newtonConjugateGradient(X, labels, classes):
    k = len(classes)
    m, n = X.shape
    all_thetas = np.zeros(shape=(k, n))
    for c in classes:
        print 'Running SGD for label', c
        y = (labels == c).astype(int)
        theta0 = np.zeros(n)
        theta = optimize.fmin_cg(cost, theta0, fprime=gradient, args=(X, y, m))
        all_thetas[c] = theta
    return all_thetas

def prediction(X, all_thetas):
    return np.argmax(sigmoid(np.dot(X, all_thetas.transpose())), axis=1)
