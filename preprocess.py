from PIL import Image
import numpy as np


def png2np(paths):
    np_arrs = []
    for path in paths:
        # np_arr shape: [1 x 784]
        np_arr = np.expand_dims(np.array(Image.open(path)).flatten(), 0)
        np_arrs.append(np_arr)
    return np.concatenate(np_arrs)


def save_png_as_np(split):
    paths = []
    with open(split + '.txt', 'r') as f:
        paths.append(f.readline())
    np_arr = png2np(paths)
    np.save(split + '.npy', np_arr)


if __name__ == '__main__':
    splits = ['train', 'test']
    for split in splits:
        save_png_as_np(split)
