#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from math import ceil
from scipy.stats import poisson
from scipy.misc import logsumexp
from numpy import logaddexp
 
 
class PoissonHMM:
    def __init__(self, prior_probs, trans_probs, rates, *, threshold=1e-2):
        self.prior_probs = np.array(prior_probs) # P_{y}
        self.trans_probs = np.array(trans_probs) # P_{ys}
        self.rates = rates # \lambda_{y}
        self.n_classes = len(rates)
 
    def _log_forward(self, X):
        sample_size = len(X)
        alpha = np.zeros(shape=(sample_size, self.n_classes))
        alpha[0] = np.log(self.prior_probs) + poisson.logpmf(x=self.X[0], mu=self.rates)
        for i in range(1, sample_size):
            alpha[i] = np.apply_along_axis(logsumexp, axis=0, arr=np.log(self.trans_probs) + alpha[i - 1]) +\
                                                poisson.logpmf(x=X[i], mu=self.rates)
 
    def _log_backward(self, X):
        sample_size = len(X)
        beta = np.zeros(shape=(sample_size, self.n_classes))
        beta[-1] = np.log(1.)
        for i in range(len(X) - 2, -1, -1):
            multiplier = logaddexp(poisson.logpmf(x=self.X[i], mu=self.rates), beta[i + 1])
            beta[i] = np.apply_along_axis(logsumexp, axis=0, arr=np.log(self.trans_probs) + multiplier)
 
    def _update_parameters(self):
        pass
 
    def fit(self, X):
        pass
 
    def predict_log_proba(self, X):
        self._log_forward(X)
        self._log_backward(X)
        Q = []
 
        alpha = np.exp(self.alpha)
        log_beta = np.exp(self.log_beta)
 
        pr = np.sum(alpha * log_beta)
 
        for c in range(self.n_classes):
            l = len(self.alpha)
            line = np.zeros(l)
            for i in range(l):
                ai = alpha[i]
                bi = log_beta[i]
                line.append(ai*bi/pr)
            Q.append(line)
        return Q
 
    def score(self, X):
        return logsumexp(self.alpha[-1])
 
    def sample(self, N):
        y = np.random.choice(np.arange(self.n_classes), size = N, replace = True, p = self.prior_probs)
        X = [poisson.rvs(self.rates[i]) for i in y]
        return X, y


def main():
    phmm = PoissonHMM([0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], [4, 12])
    X, _y = phmm.sample(1024)
    plt.hist(X, bins=10)
    plt.show()


if __name__ == "__main__":
    main()
