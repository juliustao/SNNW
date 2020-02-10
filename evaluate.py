import os

import numpy as np

from nn import Model
from helper import load_config
from datasets.mnist import save_dirs


if __name__ == '__main__':
    model_dir = 'models/mnist_1'

    model_config = load_config(os.path.join(save_dir, 'config.txt'))

    test_images = np.load(os.path.join(save_dirs.npy_dir, 'test-images.npy'))
    test_labels = np.load(os.path.join(save_dirs.npy_dir, 'test-labels.npy'))
    test_model = Model(x_input=test_images, y_true=test_labels, model_config=model_config)
    test_model.evaluate(model_dir=model_dir)
