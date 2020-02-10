import os

import numpy as np
from tqdm import tqdm

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


class Layer:
    # These layers are fully connected by default
    def __init__(self, layer_type, in_size, out_size, activation):
        # only type supported is fully connected
        self.layer_type = "fully_connected"
        # in_size is number of neuron in previous layer
        self.in_size = in_size
        # out_size is number of neurons in current layer
        self.out_size = out_size
        # activation function name
        self.activation = activation
        # make random weight array
        self.weights = np.random.normal(size=(self.in_size, self.out_size))
        # make random bias array
        self.biases = np.random.normal(size=(1, self.out_size))
        # make output array
        self.output = None


class Model:
    def __init__(self, x_input, y_true, model_config, loss_class=None):
        # x_in = [# examples x len(x)] numpy array
        self.x_input = x_input

        # y = [# examples x 1] numpy array
        self.y_true = y_true

        # the format of config is a list of dictionaries of
        # {layer_type: ..., in_size: ..., out_size: ..., activation: ...}
        # note: layers must be fully connected

        # first check that the dimensions of everything match up
        dim_error_str = "Model config dimensions do not match. (in_size of layer i != out_size of layer i-1)"
        assert (self.x_input.shape[1] == model_config[0]['in_size']), dim_error_str
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

    def forward(self, x_in, y_gt):
        # x_in shape: [1 x len(x_in)]
        y = x_in
        for layer in self.layers:
            x = np.matmul(y, layer.weights) + layer.biases
            y = name2function[layer.activation](x)
            layer.output = y
        # this should get logits array
        y_pred = y
        E = self.loss_class.function(y_pred=y_pred, y_gt=y_gt)
        return y_pred, E

    def forward_and_backward(self, x_in, y_gt, learning_rate):
        # get logits from forward pass
        y_pred, E = self.forward(x_in, y_gt)

        for i in range(0, len(self.layers), -1):
            # shape: [1 x len(logits)]
            # ??? What if the derivative depends on the actual coordinates ???
            if i == len(self.layers) - 1:
                # special case for loss layer
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
                dEdw = np.matmul(self.x_input.T, dEdx)
            else:
                dEdw = np.matmul(self.layers[i - 1].output.T, dEdx)

            # shape: [1 x len(logits)]
            dEdb = dEdx

            # adjust the weights and biases for current layer
            self.layers[i].weights = self.layers[i].weights - learning_rate * dEdw
            self.layers[i].biases = self.layers[i].biases - learning_rate * dEdb

        return y_pred, E

    def train(self, steps, learning_rate, model_dir):
        for i in tqdm(range(steps)):
            # stochastic gradient descent
            ind = self.x_input.shape[0] * np.random.uniform()
            x_in = self.x_input[ind:ind+1, :]
            y_gt = self.y_true[ind:ind+1, :]
            y_pred, loss = self.forward_and_backward(x_in, y_gt, learning_rate)
            if i % 10:
                print('Train step {}: \t\tLoss = {:8}'.format(i, loss))

        print('Finished training. Saving weights and biases...')

        weight_arr = []
        bias_arr = []
        for layer in tqdm(self.layers):
            weight_arr.append(layer.weights)
            bias_arr.append(layer.biases)
        # to access the weights of layer 3: weight_arr['arr_3']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        np.savez(os.path.join(model_dir, 'weights.npy'), *weight_arr)
        np.savez(os.path.join(model_dir, 'biases.npy'), *bias_arr)

        print('Finished saving weights and biases.')
        return

    def evaluate(self, model_dir):
        # load weights and biases
        weights_dict = np.load(os.path.join(model_dir, 'weights.npy'))
        biases_dict = np.load(os.path.join(model_dir, 'biases.npy'))
        assert (len(weights_dict) == len(self.layers)), "saved weights' shape != model weights' shape"
        assert (len(biases_dict) == len(self.layers)), "saved biases' shape != model biases' shape"

        # set model's weights and biases to the loaded weights and biases
        for i in range(len(self.layers)):
            layer = self.layers[i]
            key = 'arr_{}'.format(i)
            layer.weights = weights_dict[key]
            layer.biases = biases_dict[key]

        # evaluate for each test image
        correct = 0
        for i in range(self.x_input.shape[0]):
            x_in = self.x_input[i:i+1, :]
            y_gt = self.y_true[i:i+1, :]
            y_pred, E = self.forward(x_in, y_gt)
            if np.argmax(y_pred) == np.argmax(self.y_true[i, :]):
                correct += 1
            print('Evaluate step {}: \t\tLoss = {:8} \t\tAccuracy = {:8}'.format(i, E, correct / float(i)))
        print('Finished evaluating')