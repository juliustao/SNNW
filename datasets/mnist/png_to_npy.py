import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import save_dirs


def mnist_png_to_np(split, png_dir, npy_dir):
    with open(os.path.join(png_dir, split + '.txt'), 'r') as f:
        paths = f.read().splitlines()
    # number of labels
    labels = 0
    # doesn't matter what path we use to get the split directory name
    split_dir = os.path.dirname(os.path.dirname(paths[0]))
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
    np.save(os.path.join(npy_dir, split + '-images.npy'), np_images)
    np.save(os.path.join(npy_dir, split + '-labels.npy'), np_labels)


if __name__ == '__main__':
    png_dir = save_dirs.png_dir
    npy_dir = save_dirs.npy_dir
    splits = ['train', 'test']
    for split in splits:
        print('Current split: ' + split)
        mnist_png_to_np(split, png_dir, npy_dir)