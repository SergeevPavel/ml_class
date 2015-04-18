#!/usr/bin/env python3

from sklearn.datasets import make_classification

import numpy as np
import cvxopt as opt

class LinearSVM:
    def __init__(self, C):
        self.C = C
        self.w = None
        self.w0 = None
        self.ksi = None
        self.support_ = None

    def fit(self, X, y):
        # l -- samples count
        # n -- features count

        l, n = X.shape
        len_z = l + 1 + n

        diag = opt.matrix(0, (len_z, 1))
        diag[:n] = 1
        P = opt.spdiag(diag)

        q = opt.matrix(0, (len_z, 1), tc='d')
        q[-l:] = self.C

        h = opt.matrix(0, (2 * l, 1), tc='d')
        h[-l:] = -1

        zeros_block = opt.spmatrix([], [], [], (l, n))
        zeros_vector = opt.spmatrix([], [], [], (l, 1))
        mones_block = opt.spdiag(opt.matrix(-1, (l, 1)))
        data_block = opt.matrix(-(X.T * y).T)
        y_block = opt.matrix(y, (l, 1))

        G = opt.sparse([[zeros_block, data_block], [zeros_vector, y_block], [mones_block, mones_block]])

        result = opt.solvers.qp(P, q, G, h)
        self.w = np.array(result['x'][:n])
        self.w0 = np.array(result['x'][n])
        self.ksi = np.array(result['x'][(n + 1):])

        support_mask = np.ravel((X.dot(self.w) + self.w0) * y.reshape(l, 1) - 1 + self.ksi <= 1e-5)
        self.support_ = np.arange(l)[support_mask]
        print(self.support_)

    def descision_function(self, X):
        return X.dot(self.w) + self.w0

    def predict(self, X):
        return np.sign(self.descision_function(X))


def main():
    X, y = make_classification()
    np.place(y, y == 0, -1)
    svm = LinearSVM(1)
    svm.fit(X, y)
    print(svm.predict(X))


if __name__ == "__main__":
    main()
