#!/usr/bin/env python3

import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.cross_validation import train_test_split

class NormalLR:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        self.weights = np.linalg.solve(X.T.dot(X), X.T.dot(y))
        # print("q = {}".format(self._calc_Q(self.weights, X, y)))
        return self

    def _calc_Q(self, weights, X, y):
        return np.mean((X.dot(weights) - y) ** 2)

    def predict(self, X):
        return X.dot(self.weights)

class GradientLR(NormalLR):
    def __init__(self, *, alpha):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        self.alpha = alpha
        self.threshold = alpha / 100
        self.weights = None

    def _calc_step(self, weights, X, y):
        return X.T.dot(X.dot(weights) - y) / len(y) * self.alpha

    def fit(self, X, y):
        (l, n) = X.shape
        weights = np.random.random(n)
        while True:
            step = self._calc_step(weights, X, y)
            weights -= step
            # print("q = {}".format(self._calc_Q(weights, X, y)))
            if np.linalg.norm(step) < self.threshold:
                break
        self.weights = weights
        return self

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def sample(size, *, weights):
    X = np.ones((size, 2))
    X[:, 1] = np.random.gamma(4., 2., size)
    y = X.dot(np.asarray(weights))
    y += np.random.normal(0, 1, size)
    return X[:, 1:], y

def timed_fit(lr, data):
    start_time = time.perf_counter()
    lr.fit(*data)
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000.0

def check_mse(lr, X, y_true):
    y_pred = lr.predict(X)
    return mse(y_true, y_pred)

def test():
    glr = GradientLR(alpha=1e-5)
    nlr = NormalLR()
    for size in [128, 256, 512, 1024]:
        X, y_true = sample(size, weights=[24., 42.])
        gradient_time = timed_fit(glr, (X, y_true))
        gradient_mse = check_mse(glr, X, y_true)
        normal_time = timed_fit(nlr, (X, y_true))
        normal_mse = check_mse(nlr, X, y_true)
        print("Size = {}".format(size))
        print("Gradient: time = {:.4f}ms, mse = {}".format(gradient_time, gradient_mse))
        print("Normal: time = {:.4f}ms, mse = {}".format(normal_time, normal_mse))

def show(X, y_true, y_pred):
    plt.scatter(X, y_true)
    plt.plot(X, y_pred, color="red")
    plt.show()

def load_data():
    data = np.loadtxt("boston.csv", delimiter=',', skiprows=15)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def main():
    X, y = load_data()
    X = scale(X)
    X = np.c_[np.ones(len(y)), X]
    X_train, X_test, y_train, y_test =train_test_split(X, y)
    nlr = NormalLR()
    glr = GradientLR(alpha=1e-3)
    timed_fit(nlr, (X_train, y_train))
    timed_fit(glr, (X_train, y_train))
    nlr_y_pred = nlr.predict(X_test)
    glr_y_pred = glr.predict(X_test)
    print("NormalLR mse = {}".format(mse(y_test, nlr_y_pred)))
    print("GradientLR mse = {}".format(mse(y_test, glr_y_pred)))

if __name__ == "__main__":
    main()
