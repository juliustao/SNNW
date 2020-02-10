import numpy as np


def function(y_pred, y_gt):
    # shape y_pred [1 x _]
    # y_true is [1 x _]
    assert (abs(np.sum(y_pred) - 1.0) < 1e-6), 'y_pred fed into cross entropy is not a probability array'
    log_likelihood = -np.log(y_pred)
    loss = np.sum(log_likelihood * y_gt) / np.sum(y_gt)
    return loss


def derivative(y_pred, y_true):
    pass
