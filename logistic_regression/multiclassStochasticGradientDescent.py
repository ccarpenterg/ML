
def multiclassStochasticGradientDescent(X, classes, labels, iterations, alpha, Lambda, oscillator_factor=4):
    k = len(classes)
    m, n = X.shape
    all_thetas = np.zeros(shape=(k, n))
    for c in classes:
        y = (labels == c)
        theta, costList = stochasticGradientDescent(X, y, iterations, alpha, Lambda, oscillator_factor)
        all_thetas[c] = theta
    return all_thetas

def prediction(X, all_thetas):
    X = np.matrix(X)
    prediction = np.argmax(sigmoid(np.dot(X, all_thetas)), axis=1)
    return prediction
