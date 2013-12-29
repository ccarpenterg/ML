import numpy as np
from base import sigmoid

def cost(theta, X, y, m):
    H = sigmoid(np.dot(X, theta))
    return (-1/m) * (np.dot(y.transpose(), np.log(H)) + np.dot((1-y).transpose(), np.log(1-H)))

def gradient(theta, X, y, m):
    H = sigmoid(np.dot(X, theta))
    return (1/m) * np.dot(X.transpose(), H - y)
