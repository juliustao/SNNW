import os

import numpy as np

from nn import Model
from helper import load_config
from datasets.mnist import save_dirs


if __name__ == '__main__':
    steps = 5000
    learning_rate = 1e-6
    model_dir = 'models/mnist_1'

    model_config = load_config(os.path.join(model_dir, 'config.txt'))

    train_images = np.load(os.path.join(save_dirs.npy_dir, 'train-images.npy'))
    train_labels = np.load(os.path.join(save_dirs.npy_dir, 'train-labels.npy'))
    train_model = Model(x_input=train_images, y_true=train_labels, model_config=model_config)
    train_model.train(steps=steps, learning_rate=learning_rate, model_dir=model_dir)
