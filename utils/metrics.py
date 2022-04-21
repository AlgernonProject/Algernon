import numpy as np


def accuracy(y, y_hat):
    return np.sum(np.equal(y, y_hat)) / len(y)
