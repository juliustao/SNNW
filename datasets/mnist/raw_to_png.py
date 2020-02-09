#!/usr/bin/env python

import os
import struct

from array import array
import png
from tqdm import tqdm

import save_dirs


def read(dataset="train", path="."):
    if dataset is "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'train' or 'test'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols


def write_dataset(labels, data, size, rows, cols, output_path, dataset):
    output_dir = os.path.join(output_path, dataset)
    # create output directories
    output_dirs = [os.path.join(output_dir, str(i)) for i in range(10)]
    for dir in output_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # write data
    i = 0
    print("Writing {} dataset".format(dataset))
    for label in tqdm(labels):
        output_filename = os.path.join(output_dirs[label], str(i) + ".png")
        with open(output_filename, "wb") as h:
            w = png.Writer(cols, rows, greyscale=True)
            data_i = [data[(i*rows*cols + j*cols): (i*rows*cols + (j+1)*cols)] for j in range(rows)]
            w.write(h, data_i)
        txt_path = os.path.join(output_path, dataset + ".txt")
        with open(txt_path, "a") as f:
            f.write(output_filename + "\n")
        i += 1


if __name__ == "__main__":
    input_path = os.path.abspath(save_dirs.raw_dir)
    output_path = os.path.abspath(save_dirs.png_dir)

    for dataset in ["train", "test"]:
        labels, data, size, rows, cols = read(dataset, input_path)
        write_dataset(labels, data, size, rows, cols, output_path, dataset)
