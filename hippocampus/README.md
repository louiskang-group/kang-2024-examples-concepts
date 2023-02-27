## Overview

This directory contains
- PyTorch code that produces MF and PP encodings of memories,
- C code for simulating Hopfield networks that store these encodings,
- and PyTorch code for visualizing patterns retrieved in the Hopfield network.


## Requirements

### PyTorch

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

### C

This code requires the [Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html), which provides optimized BLAS routines for Intel CPUs. After installation, environmental variables must be set by running
```console
$ source /opt/intel/oneapi/setvars.sh

:: initializing oneAPI environment ...
...
```
Alternatively, to set these variables quietly, run
```console
$ source /opt/intel/oneapi/mkl/latest/env/vars.sh
```
Either of these commands can be added to your shell's startup files.

Successful installation and setting of environmental variables can be assessed by running `echo "${MKLROOT}"`:
```console
$ echo "${MKLROOT}"
/opt/intel/oneapi/mkl/2023.0.0
```
If an alternative BLAS library is desired, the source code can be relatively easily modified by substituting the `#include "mkl.h"` statement and replacing a few `mkl_calloc` calls with `calloc`. The compilation script `compile.sh` then must be modified to link the appropriate BLAS libraries.


## Compiling C code

On macOS, the default Clang compiler `gcc` was tested, and on Linux, the default GNU Compiler Collection `gcc` was tested.

To compile the C source code, navigate to the `src` directory and run
```console
$ ./compile.sh search.c
$ ./compile.sh dynamics.c
```

If errors are produced related to MKL libraries, try replacing the MKL-related flags in `compile.sh` with those recommended by the [Intel MKL Link Line Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html). The link flags provided were obtained by choosing the "Dynamic" linking and "Sequential" threading (i.e., no multithreading) options.


## Simulations

Pre-generated outputs for each simulation file have been provided, so they can be run in any order. To follow the simulation pipeline and regenerate outputs, run the files in sequence from `1_pathways.ipynb` to `4_visualization.ipynb`. Outputs are stored in corresponding `out_*` directories.


### `1_pathways.ipynb`

This Jupyter Python notebook generates MF and PP encodings by
- downloading the FashionMNIST dataset,
- training an autoencoder on FashionMNIST images whose middle layer represents EC,
- propagating EC encodings forward to DG, MF, and PP,
- and training a feedforward decoder to revert MF and PP encodings back to EC.


### `2a_hopfield.sh`

`2a_hopfield.sh` is a base script for running a Hopfield network simulation. At least the filename root must be provided as an argument. One can run a test simulation with
```
% ./2a_hopfield.sh test -s 20 -sparse_cue -sparse_target -theta_mid 0.5 -n_round 2 -n_value 7

STARTING SIMULATION FOR out_hopfield/test
Pattern directory contains 2048 neurons, 3 categories, and 512 examples
Sparsenseses detected: 0.020 and 0.200
...

```
