# Overview

This directory contains PyTorch code for example and concept learning in multilayer perceptrons using the DeCorr and HalfCorr loss functions.

Note that "category" or "class" in the code refers to "concept" in our manuscript.

Subdirectories:
- `converted`: Conversion script `nb2script.sh` and converted Jupyter notebooks
- `lib`: Modules used for network training
- `out`: Output files of trained networks


# Requirements

### Python packages

Required python packages are listed below, with versions checked in parentheses.
- python (3.9)
- numpy
- scipy
- pytorch (1.8.1)
- cudatoolkit (11.1.1; now superseded by pytorch-cuda)
- torchvision (0.9.1)
- matplotlib
- jupyterlab (or jupyter or a different notebook viewer)
- h5py
- ipynbname
- jupyterlab-nvdashboard (not required but helpful for monitoring multiprocessing)

### `multiprocessing` helper script `nb2script.sh`

We would like to use `torch.multiprocessing` to train replicate networks in parallel, but this library does not work from Jupyter notebooks. To provide this functionality, we provide a helper script `nb2script.sh` in the `converted` directory that converts the Jupyter notebook to a python script and discards code not required for multiprocessing. The multiprocessing script can then be called from the notebook. Remember to enable execution by running from the `converted` directory
```console
$ chmod +x nb2script.sh
```


# Training and evaluation

The following Jupyter notebooks can be evaluated in any order. In general, variable names with all capitals indicate modifiable hyperparameters.

### `1task-concept.ipynb`

This Jupyter notebook trains baseline and DeCorr networks to perform digit classification of MNIST images, which requires merging similar images into concepts.

### `1task-example.ipynb`

This Jupyter notebook trains baseline and DeCorr networks to perform set identification of MNIST images, which requires distinguishing similar images as separate examples.

### `2tasks.ipynb`

This Jupyter notebook trains baseline, DeCorr, and HalfCorr networks to perform digit classification and set identification of MNIST images.
