import os, sys
import time
import itertools

import numpy as np
import torch

import matplotlib.pyplot as plt
import h5py
import csv


def is_notebook():
    return os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py'

def relu(x):
    return x * (x > 0)

# Timing utilities
start_time = None

def start_timer():
    global start_time
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print():
    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.time()
    print("\nTotal execution time {:.3f} sec".format(end_time - start_time))


# Plot utilities
def pad(lims, margins):
    if isinstance(margins, (int, float)):
        margins = (margins, margins)
        
    lim_min = lims[0] - margins[0]*(lims[1]-lims[0])
    lim_max = lims[1] + margins[1]*(lims[1]-lims[0])
    
    return (lim_min, lim_max)

    
# multiprocessing utilities
def kwstarmap(pool, fn, kwargs_iter):
    args_for_starmap = zip(itertools.repeat(fn), kwargs_iter)
    return pool.starmap(apply_kwargs, args_for_starmap)

def apply_kwargs(fn, kwargs):
    return fn(**kwargs)

    
def plot_images(img_list, num=6, size=12):
    if isinstance(img_list, tuple):
        img_list = list(img_list)
    elif isinstance(img_list, (np.ndarray, torch.Tensor)):
        img_list = [img_list]
    elif not isinstance(img_list, list):
        raise Exception("input must be ndarray or sequence or list of ndarrays")
        
    for i in range(len(img_list)):
        if isinstance(img_list[i], torch.Tensor):
            img_list[i] = img_list[i].cpu().numpy()
            
        depth = img_list[i].ndim
        if depth == 4:
            continue
        elif depth == 3:
            img_list[i] = np.expand_dims(img_list[i], 0)
        elif depth == 2:
            img_list[i] = np.expand_dims(img_list[i], (0,1))
        else:
            raise Exception("image arrays must have depth between 2 and 4, inclusive")
            
    lengths = [imgs.shape[0] for imgs in img_list]
    if lengths.count(lengths[0]) != len(lengths):
        raise Exception("image arrays must have the same first dimension size")
    if num == -1 or num > lengths[0]:
        num = lengths[0]
    num_list = len(img_list)
    
    if num == 1:
        n_rows = 1
        n_columns = num_list
    else:
        n_rows = num_list
        n_columns = num
    ysize = size * n_rows/n_columns
    
    plt.figure(figsize=(size, ysize))
    for i, imgs in enumerate(img_list):
        if np.min(imgs) < 0.:
            imgs = 0.5 * (imgs + 1.)
        for j, img in enumerate(imgs):
            if j >= num: break
            plt.subplot(n_rows, n_columns, num*i+j+1)
            if len(img) == 1:
                plt.imshow(img[0], cmap='gray_r')
            else:
                plt.imshow(np.transpose(img, (1,2,0)))
            plt.axis('off')

    plt.show()
    
def logits_to_accuracy(logits, targets, reduction='mean'):
    accuracies = logits.argmax(-1) == targets
    if reduction == 'mean':
        return accuracies.float().mean().item()
    elif reduction == 'sum':
        return accuracies.int().sum().item()
    else:
        return accuracies.int()

