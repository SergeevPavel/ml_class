#!/usr/bin/env python3

import numpy as np
from math import sqrt


class PolyModel:
    def __init__(self, degree):
        self.n = degree + 1
        self.W = None

    def fit(self, X, Y):
        A = np.array([self._generate_features(x) for x in X], np.float64)
        b = np.array(Y, np.float64)
        # self.W = np.linalg.pinv(A).dot(b)
        # self.W = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
        self.W = np.linalg.solve(A.T.dot(A), A.T.dot(b))

    def predict(self, x):
        return np.inner(self._generate_features(x), self.W)

    def get_coefs(self):
        return self.W

    def error(self, X, Y):
        return sqrt(sum((self.predict(x) - y) ** 2 for x,y in zip(X, Y)) / len(X))

    def _generate_features(self, x):
        return np.array([x ** i for i in range(self.n)], np.float64)


def polynom_to_str(coefs):
    return "f(x)=" + "+".join("{0:.2f}*x^{1}".format(w, i) for i, w in enumerate(coefs))


def main():
    with open("learn.txt") as learn:
        learn_data = (tuple(float(x) for x in row.split()) for row in learn.readlines())
        X_learn, Y_learn = zip(*learn_data)

    with open("test.txt") as test:
        test_data = (tuple(float(x) for x in row.split()) for row in test.readlines())
        X_test, Y_test = zip(*test_data)

    MIN_DEGREE = 0
    MAX_DEGREE = 100
    learned_error = []
    test_error = []
    models = []

    for degree in range(MIN_DEGREE, MAX_DEGREE):
        model = PolyModel(degree)
        model.fit(X_learn, Y_learn)
        learned_error.append(model.error(X_learn, Y_learn))
        test_error.append(model.error(X_test, Y_test))
        models.append(model)
        print("{}: learn error = {} test error = {}".format(
                                            degree,
                                            learned_error[-1],
                                            test_error[-1]))
    optimal_degree = min(range(len(test_error)),key=test_error.__getitem__)
    print("optimal degree is {}".format(optimal_degree))
    print(polynom_to_str(models[optimal_degree].get_coefs()))


    try:
        import matplotlib.pyplot as plt
        fig1 = plt.figure()
        plt.scatter(X_test, Y_test, s=1, color='red', label="test")
        plt.scatter(X_learn, Y_learn, s=3, color='black', label="learn")
        plt.plot(X_test, [models[optimal_degree].predict(x) for x in X_test])
        plt.legend()
        fig2 = plt.figure()
        plt.plot(range(MAX_DEGREE), test_error, color='red', label="test")
        plt.plot(range(MAX_DEGREE), learned_error, color='black', label="learned")
        plt.legend()
        plt.show()
    except ImportError:
        print("matplotlib not installed...")


if __name__ == "__main__":
    main()
