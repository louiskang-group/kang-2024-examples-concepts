#!/usr/bin/env python
# coding: utf-8

# Only differences between this notebook an `1task-concept.ipynb` will be commented here. See `1task-concept.ipynb` for more thorough explanations.

# # General setup

# In[15]:


import os, sys, subprocess
import time
from glob import glob
import itertools
from functools import partial
import copy
import gc

import importlib
sys.path.insert(1, os.path.realpath('lib'))
if "utils" not in sys.modules: import utils
else: importlib.reload(utils)
if "ml" not in sys.modules: import ml
else: importlib.reload(ml)
if "train" not in sys.modules: import train
else: importlib.reload(train)


import numpy as np


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
if not utils.is_notebook():
    import torch.multiprocessing as mp


# In[16]:




# # Loading Data

# In[17]:


transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

raw_train_dataset = datasets.MNIST(
    root="../../PyTorchShared/Datasets",
    train=True,
    download=True,
    transform=transform
)

raw_test_dataset = datasets.MNIST(
    root="../../PyTorchShared/Datasets",
    train=False,
    download=True,
    transform=transform
)


# In[18]:




# # Training

# In[23]:


HIDDEN_SIZE = 100
CLASS_SIZE = 100
BATCH_SIZE = 50
NUM_SETS = 10

DECORR_STRENGTH = 0.5
NUM_REPLICATES = 16
decorr_criteria = itertools.chain.from_iterable((
    itertools.repeat(None, NUM_REPLICATES),
    itertools.repeat(ml.decorr_criterion, NUM_REPLICATES),
    itertools.repeat(ml.halfcorr_criterion, NUM_REPLICATES)
))

# Evaluate digit classification performance with held-out test data and save test
# activations and outputs.
MODE = ['testval', 'testsave']

train_dataset = (
    ml.RelabeledSubset,
    dict(dataset=raw_train_dataset, class_size=CLASS_SIZE,
         target2_config=NUM_SETS, transform=transform)
)
test_dataset = (
    ml.RelabeledSubset,
    dict(dataset=raw_test_dataset,
         target2_config='none', transform=transform)
)

model = (
    ml.TwoTargetMLP,
    dict(hidden_size=HIDDEN_SIZE,
         target1_size=10, target2_size=NUM_SETS,
         nonlinearity1=nn.Tanh(), nonlinearity2=nn.Tanh())
)

devices = itertools.cycle([torch.device('cuda', i)
                           for i in range(torch.cuda.device_count())])


# In[24]:


NUM_EPOCHS = 100
PRINT_EPOCHS = NUM_EPOCHS
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0

def NOISE_FN(data):
    mask = torch.rand_like(data)
    threshold = mask.quantile(0.2)
    mask = (mask - threshold > 0).float()
    noisy_data = data * mask
    return noisy_data



# In[25]:


kwargs_map = [dict(device=dev, decorr_criterion=param)
               for dev, param
               in zip(devices, decorr_criteria)]
kwargs_partial = dict(model=model,
                      mode=MODE,
                      train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      decorr_strength=DECORR_STRENGTH,
                      noise_fn=NOISE_FN,
                      batch_size=BATCH_SIZE,
                      learning_rate=LEARNING_RATE,
                      num_epochs=NUM_EPOCHS,
                      print_epochs=PRINT_EPOCHS)


# In[26]:




# In[27]:


if not utils.is_notebook():
    train_partial = partial(train.two_target_train, **kwargs_partial)
   
    MAX_PROCESSES = 2 * torch.cuda.device_count()
    
    if __name__ == "__main__":
        utils.start_timer()
        mp.set_start_method('spawn', force=True)
        
        num_processes = min(len(kwargs_map), MAX_PROCESSES)
        with mp.Pool(num_processes) as p:
            results = utils.kwstarmap(p, train_partial, kwargs_map)

        utils.end_timer_and_print()


# In[28]:


