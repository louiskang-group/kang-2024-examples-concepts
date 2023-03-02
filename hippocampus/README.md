# Overview

This directory contains
- PyTorch code that produces MF and PP encodings of memories,
- C code for simulating Hopfield networks that store these encodings,
- and PyTorch code for visualizing patterns retrieved in the Hopfield network.

Note that "category" or "class" in the code refers to "concept" in our manuscript.

Subdirectories:
- `out`: Neural network and Hopfield network output files
- `src`: C source code for Hopfield networks


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

### Intel MKL for C code

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

### GNU parallel for C code

Parallel execution of scripts is achieved with [GNU parallel](https://www.gnu.org/software/parallel/): O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014. It can be installed on Ubuntu with
```console
$ sudo apt install parallel
```
and on macOS with Homebrew with
```console
$ brew install parallel
```


# Compiling C code

On macOS, the default Clang compiler `gcc` was tested, and on Linux, the default GNU Compiler Collection `gcc` was tested. Remember to enable execution of scripts with `chmod +x` commands.

To compile the C source code, navigate to the `src` directory and run
```console
$ ./compile.sh search.c
$ ./compile.sh dynamics.c
```

If errors are produced related to MKL libraries, try replacing the MKL-related flags in `compile.sh` with those recommended by the [Intel MKL Link Line Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html). The link flags provided were obtained by choosing the "Dynamic" linking and "Sequential" threading (i.e., no multithreading) options.


# Simulations

Pre-generated outputs for each simulation file have been provided, so they can be run in any order. To follow the simulation pipeline and regenerate outputs, run the files in sequence from `1_pathways.ipynb` to `4_visualization.ipynb`. Outputs are stored in corresponding `out_*` directories.


### `1_pathways.ipynb`

This Jupyter notebook generates MF and PP encodings by
- downloading the FashionMNIST dataset,
- training an autoencoder on FashionMNIST images whose middle layer represents EC,
- propagating EC encodings forward to DG, MF, and PP,
- and training a feedforward decoder to revert MF and PP encodings back to EC.

### `2a_hopfield.sh`

`2a_hopfield.sh` is a base script for running a Hopfield network simulation. At least the filename root must be provided as an argument. One can run a test simulation with example load 100, sparse cues, sparse targets, and a search over theta values to maximize target overlaps:
```console
$ ./2a_hopfield.sh hopfield/test -s 100 -sparse_cue -sparse_target -theta_mid 0.5 -n_round 2 -n_value 7

STARTING SIMULATION FOR out/hopfield/test
Pattern directory contains 2048 neurons, 3 categories, and 512 examples
Sparsenseses detected: 0.020 and 0.200
...

```


### `2b_sweep.sh`

`2b_sweep.sh` is a sample script for sweeping through cue types, target types, and example loads. It calls `2a_hopfield.sh` with different parameters in parallel. It sweeps through MF examples, PP examples, and combined examples as cues. It sweeps through MF examples and PP concepts as targets. For example, to run simulations with 1, 3, 10, 30, and 100 examples stored per concept, run
```console
$ ./2b_sweep.sh 001 003 010 030 100

./2a_hopfield.sh sweep/ss-s001 -s 001 -sparse_cue -sparse_target -theta 0.5 -no_search -no_shuffle
./2a_hopfield.sh sweep/ss-s003 -s 003 -sparse_cue -sparse_target -theta 0.5 -no_search -no_shuffle
...

```


### `2c_oscillation.sh`

`2c_oscillation.sh` is a sample script for running a Hopfield network simulation with oscillatory theta. It has example load 40, oscillation period 20, and sharp changes between theta values of 0.6 and 0.2. Run
```console
$ ./2c_oscillation.sh

STARTING SIMULATION FOR out/oscillation/baseline
Pattern directory contains 2048 neurons, 3 categories, and 512 examples
Sparsenseses detected: 0.020 and 0.200
...

```


### `3_visualization.ipynb`

This Jupyter notebook generates visualizations of Hopfield network activity generated by `2b_sweep.sh` and `2c_oscillation.sh` by
- loading networks trained in `1_pathways.ipynb`
- passing Hopfield network activity through the feedforward decoder to obtain EC representations
- and passing the EC representations through the decoding layers of the autoencoder.
