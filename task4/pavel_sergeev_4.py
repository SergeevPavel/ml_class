#!/usr/bin/env python3

import random

import numpy as np
import cv2


def read_image(path):
    cv_img = cv2.imread(path, cv2.IMREAD_COLOR)
    N, M, _ = cv_img.shape
    img = cv_img[:, :, ::-1].reshape(N * M, 3)
    return img


def euclidean_distance(A, B):
    return np.sqrt(np.sum(np.square(A - B), axis=1))


def k_means(X, n_clusters, distance_metric):
    n_samples, n_features = X.shape
    clusters = np.zeros(n_clusters, n_features)
    classification = np.zeros(n_samples)
    clusters = X[random.samples(range(n_samples), n_clusters), :]

    while True:
        pass
    
    return classification, clusters


def main():
    img = read_image("superman-batman.png")
    k_means()

if __name__ == "__main__":
    main()
