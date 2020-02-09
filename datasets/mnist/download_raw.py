import os
import subprocess
import save_dirs

url = "http://yann.lecun.com/exdb/mnist/"

files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir(save_dirs.raw_dir)
    for f in files:
        subprocess.call('curl {} -O'.format(url + f).split(' '))
        subprocess.call('gunzip {}'.format(f).split(' '))
    os.chdir(cwd)
