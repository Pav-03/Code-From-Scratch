# calculate y^ = wx + y
# # error we use MSE  = 1/N (yi - y^)**2
# # to find best fitting line we have to get best W and B value 
# w use gradient descent for updating parameters
# W = W -lr.dw
# B = B - lr-db
# dw = 1/N 2 * dot x(y^-yi)
# db = 1/N 2 * dot (y^-yi)
# steps : 
# 1) training : initialise the weight and bias as zero 
#2) data point : we predict by using y^ = wx + b 
# : claculate the error
# : use gradient decsent to figure out the best [arameter and will repreat it for n time s


import numpy as np


class LinearRegression:

    def __init__(self, lr = 0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0



        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
         y_pred = np.dot(X, self.weights) + self.bias
         return y_pred