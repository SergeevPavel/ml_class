#!/usr/bin/env python3

from collections import Counter
import numpy as np

RUN_TESTS = True

def read_fasta(path):
    lst = []
    with open(path, "r") as input:
        name = None
        for row in input:
            if row.startswith(">"):
                if name is not None:
                    lst.append((name, "".join(lines)))
                name = row[1:].rstrip()
                lines = []
            else:
                lines.append(row.rstrip())
    if name is not None:
        lst.append((name, "".join(lines)))
    return lst


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    a = np.array(tuple(a))
    b = np.array(tuple(b))
    prev_line = np.arange(b.size + 1)
    for ch in a:
        curr_line = prev_line + 1
        curr_line[1:] = np.minimum(curr_line[1:],
                                      (prev_line[:-1] + (b != ch)))
        curr_line[1:] = np.minimum(curr_line[1:],
                                      curr_line[:-1] + 1)
        prev_line = curr_line
    return prev_line[-1]


def test_levenshtein():
    assert(levenshtein("", "") == 0)
    assert(levenshtein("abc","") == 3)
    assert(levenshtein("", "a") == 1)
    assert(levenshtein("abc", "abc") == 0)
    assert(levenshtein("kitten", "sitting") == 3)
    assert(levenshtein("saturday", "sunday") == 3)
    assert(levenshtein("dfas", "sdfsdfsdf") == 6)


def jaccard(s, t, n):
    S = Counter((s[i:(i + n)] for i in range(len(s) - n + 1)))
    T = Counter((t[i:(i + n)] for i in range(len(t) - n + 1)))
    intersection = sum((S & T).values())
    union = sum((S | T).values())
    return 1 - intersection / union


def jaccard_bind(n):
    return lambda s, t: jaccard(s, t, n)


def test_jaccard():
    assert(jaccard("abc", "abc", 2) == 0)
    assert(jaccard("abc", "", 2) == 1)
    assert(jaccard("kitten", "sitting", 3) == 1.0 - 1.0 / 8.0)


def lance_williams(X, dist):
    n = len(X)
    Z = np.zeros((n - 1, 3))
    D = np.zeros((2 * n - 1, 2 * n - 1))
    C_size = np.ones(2 * n - 1)
    for i in range(n):
        D[i, i] = float("+inf")
        for j in range(i + 1, n):
            D[i, j] = dist(X[i], X[j])
            D[j, i] = D[i, j]
    for i in range(n - 1):
        w = i + n
        sub_D = D[:w, :w]
        u, v = np.unravel_index(sub_D.argmin(), sub_D.shape)
        Z[i, :] = u, v, D[u, v]
        C_size[w] = C_size[u] + C_size[v]
        D[w, w] = float("+inf")
        for s in range(w):
            alpha_u = C_size[u] / C_size[w]
            alpha_v = C_size[v] / C_size[w]
            D[w, s] = alpha_u * D[u, s] + alpha_v * D[v, s]
            D[s, w] = D[w, s]
        D[(v, u), :] = float("+inf")
        D[:, (v, u)] = float("+inf")
    Z[:, (0, 1)] += 1
    return Z


def show_dendrogram(Z, **kwargs):
    from scipy.cluster.hierarchy import dendrogram, from_mlab_linkage
    from matplotlib import pyplot as plt
    dendrogram(from_mlab_linkage(Z), **kwargs)
    plt.show()


def main():
    if RUN_TESTS:
        test_levenshtein()
        test_jaccard()

    X = read_fasta("ribosome.fasta")
    L, R = tuple(zip(*X))
    # Z = lance_williams(R, jaccard_bind(1))
    # Z = lance_williams(R, jaccard_bind(8))
    # Z = lance_williams(R, jaccard_bind(16))
    Z = lance_williams(R, levenshtein)
    show_dendrogram(Z, labels=tuple(map(lambda s: s.replace(" ", "\n"), L)))


if __name__ == "__main__": 
    main()
