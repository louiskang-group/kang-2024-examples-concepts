## Overview

We share PyTorch and C code that produced our results in Kang L, Toyoizumi T. "Distinguishing examples while building concepts in hippocampal and artificial networks." *bioRxiv* (2023). [doi:10.1101/2023.02.21.529365](https://doi.org/10.1101/2023.02.21.529365).

<!--
> This GitHub repository contains code used for the manuscript. See our corresponding [Google Drive respository](https://drive.google.com/drive/folders/1TF9FIyp5DXVFqlpFIC_PlJckuJgUu9mK?usp=sharing) for large data files used for the manuscript, which must be placed in a `data` directory within the GitHub repository root.
-->

Our PyTorch code has been tested on a machine running Ubuntu 20.04 Linux with NVIDIA V100 GPUs. Our C code has been tested on machines running macOS 13 or Ubuntu 20.04 Linux with Intel CPUs.

## `hippocampus` and `ml` directories

Code concerning the biological model of hippocampal pathways to CA3 and autoassociation in CA3 (Figs. 1–3) is provided in the `hippocampus` directory. Code concerning the application of CA3-like encodings towards machine learning with the DeCorr and HalfCorr loss functions (Fig. 7) is provided in the `ml` directory. 

Further documentation is included in README files in each directory.
