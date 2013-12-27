import numpy as np
from base import sigmoid
from stochasticGradientDescent import stochasticGradientDescent

def multiclassStochasticGradientDescent(X, labels, classes, iterations, alpha, Lambda, oscillator_factor=4):
    k = len(classes)
    m, n = X.shape
    all_thetas = np.zeros(shape=(k, n))
    for c in classes:
        print 'Running SGD for label', c
        y = (labels == c).map(int)
        theta, costList = stochasticGradientDescent(X, y, iterations, alpha, Lambda, oscillator_factor)
        all_thetas[c] = theta
    return all_thetas

def prediction(X, all_thetas):
    X = np.matrix(X)
    prediction = np.argmax(sigmoid(np.dot(X, all_thetas.transpose())), axis=1)
    prediction = np.squeeze(np.asarray(prediction))
    return prediction

def accuracy(y, prediction):
    accur = (y == prediction)
    accur = accur.astype(int)
    return (float(accur.sum()) / accur.size) * 100
