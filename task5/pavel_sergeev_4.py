#!/usr/bin/env python3

import random

from sklearn import preprocessing
import numpy as np


def read_data(path):
    X, y = np.split(np.genfromtxt(path, delimiter=','), [-1], axis=1)
    y = (y * 2 - 1).ravel()
    m = X.shape[0]
    X = preprocessing.scale(X)
    X = np.c_[-np.ones(m), X]
    return X, y


def log_loss(M):
    L = np.log2(1 + np.exp(-M))
    dL = -np.exp(-M) / ((1 + np.exp(-M)) * np.log(2))
    return L, dL


def sigmoid_loss(M):
    S = 2 / (1 + np.exp(M))
    dS = -2 * np.exp(M) / (1 + np.exp(M)) ** 2
    return S, dS


class GradientDescent:
    def __init__(self, *, alpha, threshold=1e-2, loss=sigmoid_loss):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        if threshold <= 0:
            raise ValueError("threshold should be positive")
        self.alpha = alpha
        self.threshold = threshold
        self.loss = loss
        self.weights = None

    def fit(self, X, y):
        errors = []
        n = X.shape[1]
        weights = np.random.uniform(-1.0 / (2.0 * n), 1.0 / (2.0 * n), size=n)
        while True:
            predictions = X.dot(weights.T)
            L, dL = self.loss(predictions * y)
            dQ = np.sum((X.T * dL * y).T, axis=0)
            err = np.sum(L)
            errors.append(err)
            dw = -self.alpha * dQ
            if np.linalg.norm(dw) < self.threshold:
                break
            weights += dw
        self.weights = weights
        return errors

    def predict(self, X):
        return np.sign(X.dot(self.weights.T))


class SGD:
    def __init__(self, *, alpha, loss=sigmoid_loss, k=1, n_iter=100):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        self.k = k
        self.n_iter = n_iter
        self.alpha = alpha
        self.loss = loss
        self.weights = None

    def fit(self, X, y):
        if X.shape[0] < self.k:
            raise ValueError("samples count must be greater than k")
        errors = []
        n = X.shape[1]
        eta = 1 / len(X)
        weights = np.random.uniform(-1.0 / (2.0 * n), 1.0 / (2.0 * n), size=n)
        predictions = X.dot(weights.T)
        L, _ = self.loss(predictions * y)
        Q = np.sum(L)
        for it in range(self.n_iter):
            mask = np.random.choice(X.shape[0], self.k, replace=False)
            sub_X = X[mask, :]
            sub_y = y[mask]
            predictions = sub_X.dot(weights.T)
            L, dL = self.loss(predictions * sub_y)
            dQ = np.sum((sub_X.T * dL * sub_y).T, axis=0)
            eps = np.sum(L)
            Q = (1 - eta) * Q + eta * eps
            errors.append(Q)
            dw = -self.alpha * dQ
            weights += dw
        self.weights = weights
        return errors

    def predict(self, X):
        return np.sign(X.dot(self.weights.T))


def print_precision_recall(classes, y_pred, y_test):
    for c in classes:
        tp = sum((1 if pred == c and test == c else 0
                        for pred, test in zip(y_pred, y_test)))
        fp = sum((1 if pred == c and test != c else 0
                        for pred, test in zip(y_pred, y_test)))
        fn = sum((1 if pred != c and test == c else 0
                        for pred, test in zip(y_pred, y_test)))
        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            print("c = {c} precision = {precision:.3f} recall = {recall:.3f}".format(**locals()))
        except ZeroDivisionError:
            print("tp = {tp} fp = {fp} fn = {fn}".format(**locals()))
    print("-----------")


def train_test_split(X, y, ratio):
    shuffled = list(zip(X, y))
    random.shuffle(shuffled)
    count = round(len(shuffled) * ratio)
    X_train, y_train = zip(*shuffled[0:count])
    X_test, y_test = zip(*shuffled[count:])
    # assert(len(X_train) / (len(X_test) + len(X_train)) == ratio)
    # assert(len(y_train) / (len(y_test) + len(y_train)) == ratio)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def test_algorithms(X, y):
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)

    gd = GradientDescent(alpha=1e-3, loss=sigmoid_loss)
    gd.fit(X_train, y_train)
    y_pred = gd.predict(X_test)
    print("gd, sigmoid")
    print_precision_recall([-1, 1], y_pred, y_test)

    gd = GradientDescent(alpha=1e-3, loss=log_loss)
    gd.fit(X_train, y_train)
    y_pred = gd.predict(X_test)
    print("gd, log")
    print_precision_recall([-1, 1], y_pred, y_test)

    sgd = SGD(alpha=1e-3, loss=sigmoid_loss, n_iter=1000)
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    print("sgd sigmoid")
    print_precision_recall([-1, 1], y_pred, y_test)

    sgd = SGD(alpha=1e-3, loss=log_loss, n_iter=1000)
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    print("sgd log")
    print_precision_recall([-1, 1], y_pred, y_test)


def plot_graphics():



def main():
    PIMA_DATA = "pima-indians-diabetes.csv"
    X, y = read_data(PIMA_DATA)
    test_algorithms(X, y)

if __name__ == "__main__":
    main()
