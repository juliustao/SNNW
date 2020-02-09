import numpy as np
import os
from Layer import Layer
import losses
import activations

name2function = {
    'ReLu': activations.relu.function,
    'Sigmoid': activations.sigmoid.function,
    'Softmax': activations.softmax.function,
    'Tanh': activations.tanh.function,
}

name2derivative = {
    'ReLu': activations.relu.derivative,
    'Sigmoid': activations.sigmoid.derivative,
    # 'Softmax': activations.Softmax.derivative,
    'Tanh': activations.tanh.derivative,
}


class Model:
    def __init__(self, x_in, y_true, model_config, loss_class=None):
        # x_in = [# examples x len(x)] numpy array
        self.x_in = x_in

        # y = [# examples x 1] numpy array
        self.y_true = y_true

        # the format of config is a list of dictionaries of
        # {layer_type: ..., in_size: ..., out_size: ..., activation: ...}
        # note: layers must be fully connected

        # first check that the dimensions of everything match up
        dim_error_str = "Model config dimensions do not match. (in_size of layer i != out_size of layer i-1)"
        assert (self.x_in.shape[1] == model_config[0]['in_size']), dim_error_str
        for i in range(1, len(model_config)):
            assert (model_config[i-1]['out_size'] == model_config[i]['in_size']), dim_error_str

        # check that the last layer has activation softmax
        activation_error_str = 'Only softmax and cross-entropy are supported for the last layer and loss, respectively'
        assert (model_config[-1]['activation'] == 'softmax'), activation_error_str

        # initialize model's list of layers
        self.layers = []
        for layer_config in model_config:
            layer = Layer(layer_type=layer_config['layer_type'],
                          in_size=layer_config['in_size'],
                          out_size=layer_config['out_size'],
                          activation=layer_config['activation'])
            self.layers.append(layer)

        # only loss supported is cross-entropy
        self.loss_class = losses.CrossEntropy

    def forward(self, x_in, y_true):
        # x_in shape: [1 x len(x_in)]
        y = x_in
        for layer in self.layers:
            x = np.matmul(y, layer.weights) + layer.biases
            y = name2function[layer.activation](x)
            layer.output = y
        # this should get logits array
        y_pred = y
        E = self.loss_class.function(y_pred=y_pred, y_true=y_true)
        return y_pred, E

    def forward_and_backward(self, x_in, y_true, learning_rate):
        # get logits from forward pass
        y_pred, E = self.forward(x_in, y_true)

        for i in range(0, len(self.layers), -1):
            # shape: [1 x len(logits)]
            # ??? What if the derivative depends on the actual coordinates ???
            if i == len(self.layers) - 1:
                # special case for loss layer
                # dEdy = name2derivative[self.loss_fn](y_pred=y_pred, y_true=self.y_true)
                # We only have Cross-Entropy loss, so just calculate derivative of loss with respect to x before softmax
                dEdx = self.layers[i].output - y_pred
            else:
                dEdy = np.matmul(dEdx, self.layers[i+1].weights.T)
                # shape: [1 x len(logits)]
                # Hadamard of dy/dx and dE/dy = dE/dx
                dEdx = name2derivative[self.layers[i].activation](self.layers[i].output) * dEdy

            # shape: [len(prev_layer) x len(logits)]
            if i == 0:
                # special case for input layer
                dEdw = np.matmul(self.x_in.T, dEdx)
            else:
                dEdw = np.matmul(self.layers[i - 1].output.T, dEdx)

            # shape: [1 x len(logits)]
            dEdb = dEdx

            # adjust the weights and biases for current layer
            self.layers[i].weights = self.layers[i].weights - learning_rate * dEdw
            self.layers[i].biases = self.layers[i].biases - learning_rate * dEdb

        return y_pred, E

    def train(self, steps, learning_rate):
        total = self.x_in.shape[0]
        correct = 0
        for i in range(steps):
            ind = total * np.random.uniform()
            x = self.x_in[ind:ind+1, :]
            y_pred, loss = self.forward_and_backward(x, learning_rate)
            if np.argmax(y_pred) == np.argmax(self.y_true[ind, :]):
                correct += 1
            if i % 10:
                print('Train step {}: Loss = {:8}'.format(i, loss))
        accuracy = correct / float(total)
        print('Accuracy: {:8}'.format(accuracy))

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights.npz')
        weight_arr = []
        for layer in self.layers:
            weight_arr.append(layer.weights)
        # to access the weights of layer 3: weight_arr['arr_3']
        np.savez(path, *weight_arr)
        return

    def evaluate(self):
        pass
