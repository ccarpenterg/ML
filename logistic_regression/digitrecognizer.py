import sys
import numpy as np
from multiclassStochasticGradientDescent import multiclassStochasticGradientDescent, prediction, accuracy
from base import trainingExamples

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if __name__ == '__main__':

    df, X, y, Xval, yval = trainingExamples(sys.argv[1], 10000, "label", "pixel") 
    all_thetas = multiclassStochasticGradientDescent(X, y, classes, 3, 0.01, 0)
    print all_thetas
    trainingPrediction = prediction(X, all_thetas)
    trainingAccuracy = accuracy(np.array(y), trainingPrediction)
    print np.array(y)
    print trainingPrediction
    print 'Training accuracy (%):', trainingAccuracy
