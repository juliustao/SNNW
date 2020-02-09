import numpy as np
from Layer import Layer
import losses
import activations

name2function = {
    'ReLu': activations.ReLu.function,
    'Sigmoid': activations.Sigmoid.function,
    'Softmax': activations.Softmax.function,
    'Tanh': activations.Tanh.function,
    'CrossEntropy': losses.CrossEntropy.function
}

name2derivative = {
    'ReLu': activations.ReLu.derivative,
    'Sigmoid': activations.Sigmoid.derivative,
    'Softmax': activations.Softmax.derivative,
    'Tanh': activations.Tanh.derivative,
    'CrossEntropy': losses.CrossEntropy.derivative
}


class Model:
    def __init__(self, x_in, y_true, model_config, loss_fn, learning_rate):
        # x_in = [1 x len(x)] numpy array
        self.x_in = x_in

        # y = [1 x len(y)] numpy array
        self.y_true = y_true

        # the format of config is a list of dictionaries of
        # {layer_type: ..., in_size: ..., out_size: ..., activation: ...}
        # note: layers must be fully connected

        # first check that the dimensions of everything match up
        dim_error_str = "Model config dimensions do not match. (in_size of layer i != out_size of layer i-1)"
        assert (x_in.shape[1] == model_config[0]['in_size']), dim_error_str
        for i in range(1, len(model_config)):
            assert (model_config[i-1]['out_size'] == model_config[i]['in_size']), dim_error_str

        # initialize model's list of layers
        self.layers = []
        for layer_config in model_config:
            layer = Layer(layer_type=layer_config['layer_type'],
                          in_size=layer_config['in_size'],
                          out_size=layer_config['out_size'],
                          activation=layer_config['activation'])
            self.layers.append(layer)

        # only losses function supported is cross-entropy
        self.loss_fn = 'CrossEntropy'

        # learning rate
        self.learning_rate = learning_rate

    def forward(self):
        y = self.x_in
        for layer in self.layers:
            x = np.matmul(y, layer.weights) + layer.biases
            y = name2function[layer.activation](x)
            layer.output = y
        # this should return logits array
        return y

    def forward_and_backward(self):
        # get logits from forward pass
        y_pred = self.forward()

        # shape: [1 x len(logits)]
        E = name2function[self.loss_fn](y_pred=y_pred, y_true=self.y_true)

        for i in range(0, len(self.layers)-1, -1):
            # shape: [1 x len(logits)]
            # ??? What if the derivative depends on the actual coordinates???
            if i == len(self.layers)-1:
                dEdy = name2derivative[self.loss_fn](y_pred=y_pred, y_true=self.y_true)
            else:
                dEdy = np.matmul(dEdx, self.layers[i+1].weights.T)

            # shape: [1 x len(logits)]
            # Hadamard of dy/dx and dE/dy = dE/dx
            dEdx = name2derivative[self.layers[i].activation](self.layers[i].output) * dEdy

            # shape: [len(prev_layer) x len(logits)]
            if i == 0:
                dEdw = np.matmul(self.layers[i-1].output.T, dEdx)
            else:
                dEdw = np.matmul(self.x_in.T, dEdx)

            # shape: [1 x len(logits)]
            dEdb = dEdx

            # adjust the weights and biases for current layer
            self.layers[i].weights = self.layers[i].weights - self.learning_rate * dEdw
            self.layers[i].biases = self.layers[i].biases - self.learning_rate * dEdb

        return E

    def train(self, steps):
        losses = np.zeros(steps)
        for i in range(steps):
            losses[i] = self.forward_and_backward()
        return losses
