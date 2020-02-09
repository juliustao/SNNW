import numpy as np


def function(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative(x):
    sigmoid = function(x)
    return sigmoid * (1 - sigmoid)
