## Overview

We share our C code for simulating Hopfield networks and our PyTorch code for transforming memories along hippocampal pathways and for exploring the HalfCorr loss function. Our results are presented in Kang L, Toyoizumi T. "Distinguishing examples while building concepts in hippocampal and artificial networks." *bioRxiv* (2023). [doi:10.1101/2023.02.21.529365](https://doi.org/10.1101/2023.02.21.529365).

> This GitHub repository contains code used for the manuscript. See our corresponding [Google Drive respository](https://drive.google.com/drive/folders/1TF9FIyp5DXVFqlpFIC_PlJckuJgUu9mK?usp=sharing) for data files used for the manuscript.




## 1_pathways: Transforming memories from images to CA3 and back

### Installation

Required python packages are listed below, with versions checked in parentheses.
- python (3.9)
- numpy
- scipy
- pytorch (1.8.1)
- cudatoolkit (11.1.1; now superseded by pytorch-cuda)
- torchvision (0.9.1)
- matplotlib
- h5py