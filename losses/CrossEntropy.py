import numpy as np


def function(y_pred, y_true):
    # shape y_pred [1 x _]
    # y_true is [1 x _]
    assert (abs(np.sum(y_pred) - 1.0) < 1e-6), 'y_pred fed into CrossEntropy is not a probability array'
    log_likelihood = -np.log(y_pred)
    loss = np.dot(log_likelihood, y_true) / np.sum(y_true)
    return loss


def derivative(y_pred, y_true):
    pass
