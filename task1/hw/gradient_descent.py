#!/usr/bin/env python3

from math import sqrt


def norm(lst):
    return sqrt(sum(x ** 2 for x in lst))


class PolyModel:
    def __init__(self, degree):
        self.n = degree + 1

    def fit(self, X, Y):
        self.W = [0.0 for i in range(self.n)]
        self.mu, self.sigma = self._calc_norm_params(X)
        self.X_features = [self._get_norm_features(x) for x in X]
        eps = 0.1
        alpha = 0.000001
        while True:
            grad = self._grad(self.X_features, Y)
            if norm(grad) < eps:
                break
            for i in range(self.n):
                self.W[i] += -alpha * grad[i]
            # print("err={} grad={}".format(self._error(self.X_features, Y), norm(grad)))
            print("grad = {}".format(norm(grad)))

    def _grad(self, X_features, Y):
        grad = [0 for j in range(self.n)]
        for features, y in zip(X_features, Y):
            for j in range(self.n):
                grad[j] += + 2 * (self._predict(features) - y) * features[j]
        return grad

    def _get_norm_features(self, x):
        return [(x ** i - self.mu[i]) / self.sigma[i] for i in range(self.n)]

    def _calc_norm_params(self, X):
        mu_params = [0,]
        sigma_params = [1,]
        for i in range(1, self.n):
            mu = 0
            minimum = float('+inf')
            maximum = float('-inf')
            for x in X:
                feature = x ** i
                minimum = min(minimum, feature)
                maximum = max(maximum, feature)
                mu += feature
            mu /= len(X)
            mu_params.append(mu)
            sigma_params.append(maximum - minimum)
        return mu_params, sigma_params

    def _predict(self, features):
        return sum(f * w for f, w in zip(features, self.W))

    def _error(self, X_features, Y):
        err = 0
        for features, y in zip(X_features, Y):
            err += (self._predict(features) - y) ** 2
        return err

    def predict(self, x):
        return self._predict(self._get_norm_features(x))

    def error(self, X, Y):
        X_features = [self._get_norm_features(x) for x in X]
        return self._error(X_features)


def main():
    with open("learn.txt") as learn:
        learn_data = (tuple(float(x) for x in row.split()) for row in learn.readlines())
        X_learn, Y_learn = zip(*learn_data)

    with open("test.txt") as test:
        test_data = (tuple(float(x) for x in row.split()) for row in test.readlines())
        X_test, Y_test = zip(*test_data)

    model = PolyModel(50)
    model.fit(X_learn, Y_learn)

    try:
        import matplotlib.pyplot as plt
        plt.scatter(X_test, Y_test, s=1, color='red', label="test")
        plt.scatter(X_learn, Y_learn, s=1, color='black', label="learn")
        plt.legend()
        plt.show()
    except ImportError:
        print("matplotlib not installed...")


if __name__ == "__main__":
    main()
