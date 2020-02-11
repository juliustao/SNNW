# Slowest-Neural-Net-in-the-West
My custom neural net architecture written in Python to 
practice my ML skills.


## Install
Make a new Python virtual environment (version >= 3.6).
Install this project with pip. In terminal, run

`pip3 install SNNW`

## Download MNIST dataset
Open a new Python shell or Jupyter Notebook and run

`import SNNW`

Set `raw_dir = ` the directory where you want to
save the MNIST raw data files. Run

`SNNW.dataset.mnist.download_raw(raw_dir)`

Set `png_dir = ` the directory where you want to
save the MNIST .png images and path text files. Run

`SNNW.dataset.mnist.raw_to_png(raw_dir, png_dir)`

Set `npy_dir = ` the directory where you want to
save the MNIST .npy image and label numpy arrays. Run

`SNNW.dataset.mnist.png_to_npy(png_dir, npy_dir)`

## Get model config
Set `config_path = ` the path to where you want to save
the training/testing model's config file. Run

`SNNW.nn.config.get_config_1(config_path)`
or
`SNNW.nn.config.get_config_2(config_path)`

to write a sample config file to `config_path`,
or write your own config file and place it where
`config_path` points to.

If you write a custom config file, make sure that it
follows the given format to prevent run-time errors!


## Train model
Set `model_dir = ` the directory where the trained model's
weights and biases will be stored.

Set `train_image_path = ` the path to where the .npy file for
the training image arrays are located.
This file should be located inside the `npy_dir`
you specified earlier. 

Set `train_label_path = ` the path to where the .npy file for
the training label arrays are located.
This file should be located inside the `npy_dir`
you specified earlier.

Set `steps = ` the number of training steps you would
like to train for. The default is `60,000`.

Set `learning_rate = ` the learning rate you would like to 
train with. The default is `5e-4`.

Run `SNNW.train(model_dir, config_path, train_image_path,
train_label_path, steps, learning_rate)`.

## Test model

Set `test_image_path = ` the path to where the .npy file for
the training image arrays are located.
This file should be located inside the `npy_dir`
you specified earlier. 

Set `test_label_path = ` the path to where the .npy file for
the training label arrays are located.
This file should be located inside the `npy_dir`
you specified earlier.

Run `SNNW.evaluate(model_dir, config_path, test_image_path,
test_label_path)`.
