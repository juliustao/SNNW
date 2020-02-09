import os

this_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))

raw_dir = os.path.join(this_dir, 'raw')
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

png_dir = os.path.join(this_dir, 'png')
if not os.path.exists(png_dir):
    os.makedirs(png_dir)

npy_dir = os.path.join(this_dir, 'npy')
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)
