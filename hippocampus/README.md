## Overview

This directory contains
- PyTorch code that produces MF and PP encodings of memories,
- C code for simulating Hopfield networks that store these encodings,
- and PyTorch code for visualizing patterns retrieved in the Hopfield network.


## Requirements

This code has been tested on machines with Intel CPUs running macOS 13 and Ubuntu Linux 20.04. On macOS, the default Clang compiler is used, and on Linux, the default GNU Compiler Collection is used.

This code requires the [Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html), which provides optimized BLAS routines for Intel CPUs. Successful installation and setup of environmental variables can be assessed by running `echo "${MKLROOT}"`:
```console
$ echo "${MKLROOT}"
/opt/intel/oneapi/mkl/2021.3.0
```
If an alternative BLAS library is desired, the source code can be relatively easily modified by substituting the `#include "mkl.h"` statement and replacing a few `mkl_calloc` calls with `calloc`. The compilation script `compile.sh` then must be modified to link the appropriate BLAS libraries.


Required python packages are listed below, with versions checked in parentheses.
- python (3.9)
- numpy
- scipy
- pytorch (1.8.1)
- cudatoolkit (11.1.1; now superseded by pytorch-cuda)
- torchvision (0.9.1)
- matplotlib
- h5py

## Compilation

To compile the C source code, navigate to the `src` directory and run
```console
$ ./compile.sh search.c
$ ./compile.sh dynamics.c
```
If errors are produced related to MKL libraries, try replacing the MKL-related flags in `compile.sh` with those recommended by the [Intel MKL Link Line Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html). The link flags provided were obtained by choosing the "Single Dynamic Library" option.


## `1_sim.sh` and `1_sweep.sh`

`1_sim.sh` is a base script for running a Hopfield network simulation. At least the filename root must be provided as an argument. One can run a test simulation with
```
$ ./1_sim.sh 

```
