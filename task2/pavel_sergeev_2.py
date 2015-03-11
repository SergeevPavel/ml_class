#!/usr/bin/env python3

import random
import sys
from collections import defaultdict
from math import sqrt


def load_data(file_name):
    X = []
    y = []
    with open(file_name, "r") as input:
        for row in input:
            if not row.startswith("#"):
                splited = row.split(",")
                y.append(int(splited[0]))
                X.append([float(x) for x in splited[1:]])
    return X, y


def train_test_split(X, y, ratio):
    shuffled = list(zip(X, y))
    random.shuffle(shuffled)
    count = round(len(shuffled) * ratio)
    X_train, y_train = zip(*shuffled[0:count])
    X_test, y_test = zip(*shuffled[count:])
    # assert(len(X_train) / (len(X_test) + len(X_train)) == ratio)
    # assert(len(y_train) / (len(y_test) + len(y_train)) == ratio)
    return X_train, y_train, X_test, y_test


def euclidean_distance(a, b):
    return sqrt(sum([(xa - xb) ** 2 for xa, xb in zip(a, b)]))


def manhattan_distance(a, b):
    return sum([abs(xa - xb) for xa, xb in zip(a, b)])


def knn(X_train, y_train, X_test, k, dist):
    y_test = []
    for features in X_test:
        neighbors = sorted(zip(X_train, y_train),
                            key=lambda sample: dist(sample[0], features))
        vote = defaultdict(int)
        for _, y in neighbors[:k]:
            vote[y] += 1
        y_test.append(max(vote, key=vote.get))
    return y_test


def print_precision_recall(y_pred, y_test):
    n_classes = len(set(y_test))
    for c in range(1, n_classes + 1):
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


def loocv(X_train, y_train, dist):
    min_error = sys.maxsize
    opt_k = 0
    for k in range(1, len(X_train) - 1):
        error = 0
        for i in range(len(X_train)):
            X_samples, y_samples = list(X_train), list(y_train)
            X, y = X_samples[i], y_samples[i]
            del X_samples[i]
            del y_samples[i]
            error += (knn(X_samples, y_samples, [X], k, dist)[0] != y)
        # print("k={k} error={error}".format(**locals()))
        if error <= min_error:
            min_error = error
            opt_k = k
    return opt_k


def main():
    X, y = load_data("wine.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)

    euclidian_opt_k = loocv(X_train, y_train, euclidean_distance)
    print("euclidian optimal k = {}".format(euclidian_opt_k))
    y_pred = knn(X_train, y_train, X_test, euclidian_opt_k, euclidean_distance)
    print_precision_recall(y_pred, y_test)

    manhattan_opt_k = loocv(X_train, y_train, manhattan_distance)
    print("manhattan optimal k = {}".format(manhattan_opt_k))
    y_pred = knn(X_train, y_train, X_test, manhattan_opt_k, manhattan_distance)
    print_precision_recall(y_pred, y_test)



if __name__ == "__main__":
    main()
