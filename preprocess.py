import os
from tqdm import tqdm
from PIL import Image
import numpy as np


def save_png_as_np(split):
    with open(split + '.txt', 'r') as f:
        paths = f.read().splitlines()
    # number of labels
    labels = 0
    # doesn't matter what path we use to get the split directory name
    split_dir = os.path.basename(os.path.dirname(os.path.dirname(paths[0])))
    for _, dirnames, _ in os.walk(split_dir):
        labels += len(dirnames)

    np_images = []
    np_labels = []
    for path in tqdm(paths):
        # np_image shape: [1 x 784]
        np_image = np.expand_dims(np.array(Image.open(path)).flatten(), 0)
        np_images.append(np_image)

        # get label from folder name of file
        label = int(os.path.basename(os.path.dirname(path)))

        # np_label shape: [1 x #labels]
        np_label = np.zeros(shape=(1, labels))
        np_label[0, label] = 1.0
        np_labels.append(np_label)
    np.save(split + '-images.npy', np_images)
    np.save(split + '-labels.npy', np_labels)


if __name__ == '__main__':
    # assumes that train.txt and test.txt are in the same directory
    splits = ['test']
    for split in splits:
        print('Current split: ' + split)
        save_png_as_np(split)
