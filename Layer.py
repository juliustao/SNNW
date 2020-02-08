import numpy as np


class Layer:
    # These layers are fully connected by default
    def __init__(self, layer_type, in_size, out_size, activation):
        # only type supported is fully connected
        self.layer_type = "fully_connected"
        # in_size is number of neuron in previous layer
        self.in_size = in_size
        # out_size is number of neurons in current layer
        self.out_size = out_size
        self.activation = activation
        # make weight list
        self.weights = np.random.normal(size=(self.in_size, self.out_size))
        # make bias list
        self.biases = np.random.normal(size=(1, self.out_size))


