# Slowest-Neural-Net-in-the-West
My custom neural net architecture written in Python to 
practice my ML skills.

## Requirements
I successfully ran this project on Python 3.6.9,
but I'm sure that any Python version >= 3.6 works.
The Python packages required are in the requirements.txt.

Please create a new Python virtual environment to run
this project in so there are no dependency conflicts.

## TL;DR How-To
In terminal, navigate to this project's directory and run:
```
python3 datasets/mnist/download_raw.py
python3 datasets/mnist/raw_to_png.py
python3 datasets/mnist/png_to_npy.py
python3 train.py
python3 evaluate.py
```

## Folders

### activations/
This folder contains the functions and derivatives for 
the following activation functions:
1. relu
2. sigmoid
3. softmax
4. tanh

Currently, only the above activation functions are supported.

### datasets/
This folder contains scripts to download and process the 
MNIST dataset. Navigate to the datasets/mnist directory and 
execute the following commands:
1. `python3 download_raw.py`
2. `python3 raw_to_png.py`
3. `python3 png_to_npy.py`

### losses/
This folder contains the functions and derivatives for
the following loss functions:
1. cross-entropy

Currently, only the above activation functions are supported. 

### models/
This folder contains a folder for each model.

Each model's folder contains a config.txt file that specifies
how to build that model.
To create your own custom config.txt file, just follow
the format of the config.txt files that are included.

After training, the models' weights and biases
will be saved to the model's folder.

## Files

### .gitignore
The file that specifies what files/folders to ignore
when adding and committing with git.

### evaluate.py
Specify the directory of the model you would like to evaluate
on in the evaluate.py file.

*Note:* The folder must contain the 
`config.txt`, `weights.npz`, and`biases.npz`files.
After specifying, you can run this file with the command
`python3 evaluate.py`. 

### helper.py
This file contains a helper method to load the `config.txt`
files found in a model's directory.

### LICENSE
MIT License. Included to appease lawyers.

### nn.py
This file contains the Model and Layer class that are
inherited from to create Model and Layer objects used
during training and evaluation.

### requirements.txt
This file specifies the Python packages and versions
that are required to run this project.

If you `git clone` this project, you can run
`pip install -r requirements.txt` to install all of the
dependencies in the requirements.txt file.

### setup.py
This is the setup script used for packaging this project.

### train.py
Specify the directory of the model you would like to evaluate
on in the evaluate.py file.

*Note:* The folder must contain a `config.txt` file.
After specifying, you can run this file with the command
`python3 evaluate.py`.

The weights and biases of the trained model will be saved
in the model directory that was specified.
