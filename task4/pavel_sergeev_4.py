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
    classification = np.zeros(n_samples)
    clusters = np.random.random_integers(low=0, high=255, size=(n_clusters, n_features))
    distance = np.zeros((n_clusters, n_samples))

    while True:
        for i, c in enumerate(clusters):
            distance[i] = distance_metric(X, c)
        new_classification = np.argmin(distance, axis=0)
        print(np.sum(new_classification != classification))
        if np.sum(new_classification != classification) == 0:
            break
        classification = new_classification
        for i in range(n_clusters):
            mask = classification == i
            clusters[i] = np.sum(X[mask], axis=0) / np.sum(mask)
    return classification, clusters


def centroid_histogram(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return counts


def plot_colors(hist, centroids):
    bar = np.zeros((50, 500, 3), dtype=np.uint8)
    start_x = 0
    sum_hist = np.sum(hist)
    for (percent, color) in zip(hist, centroids):
        end_x = start_x + percent * 500 / sum_hist
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50),
        color.astype("uint8").tolist()[::-1], -1)
        start_x = end_x
    return bar


def recolor(image, kmeans, n_colors):
    labels, clusters = kmeans
    N, M, _ = image.shape
    for i in range(N):
        for j in range(M):
            c = labels[i * M + j]
            image[i, j] = clusters[c][::-1]
    return image


def main():
    n_colors = 16
    path = "superman-batman.png"

    img = read_image(path)
    base_img = cv2.imread(path, cv2.IMREAD_COLOR)
    labels, clusters = k_means(img, n_colors, euclidean_distance)
    hist = centroid_histogram(labels)
    bar = plot_colors(hist, clusters)
    result = recolor(base_img, (labels, clusters), n_colors)
    cv2.imshow("bar", bar)
    cv2.imshow("result", result)
    cv2.imwrite("result.png", result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
