#!/usr/bin/env python
# coding: utf-8

# # General setup

# In[3]:


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


# In[4]:




# # Loading Data

# In[5]:


transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

raw_train_dataset = datasets.MNIST(
    root="../../../PyTorchShared/Datasets",
    train=True,
    download=True,
    transform=transform
)

raw_test_dataset = datasets.MNIST(
    root="../../../PyTorchShared/Datasets",
    train=False,
    download=True,
    transform=transform
)


# In[6]:




# # Training

# In[7]:


HIDDEN_SIZE = 50
CLASS_SIZES = [200, 400, 600, 800, 1000]
num_sizes = len(CLASS_SIZES)
BATCH_SIZE = 50

DECORR_STRENGTH = 0.5
NUM_REPLICATES = 32

class_sizes = [rep
               for size in CLASS_SIZES
               for rep in (size,)*NUM_REPLICATES]*2
decorr_criteria = [rep
                   for criterion in (None, ml.decorr_criterion)
                   for rep in (criterion,)*NUM_REPLICATES*num_sizes]

MODE = 'testval'

train_datasets = [(
    ml.RelabeledSubset,
    dict(dataset=raw_train_dataset, class_size=class_size,
         target2_config='none', transform=transform)
) for class_size in class_sizes]
test_dataset = (
    ml.RelabeledSubset,
    dict(dataset=raw_test_dataset,
         target2_config='none', transform=transform)
)

model = (
    ml.MLP,
    dict(hidden_size=HIDDEN_SIZE, target_size=10,
         nonlinearity1=nn.Tanh(), nonlinearity2=nn.Tanh())
)

devices = itertools.cycle([torch.device('cuda', i)
                           for i in range(torch.cuda.device_count())])


# In[8]:


NUM_EPOCHS = 40
PRINT_EPOCHS = 0
LR_MILESTONES = [NUM_EPOCHS]
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
NOISE_FN = None


# In[9]:


kwargs_map = [dict(device=dev,
                   train_dataset=param0,
                   decorr_criterion=param1)
               for dev, param0, param1
               in zip(devices,
                      train_datasets,
                      decorr_criteria)]
kwargs_partial = dict(model=model,
                      mode=MODE,
                      test_dataset=test_dataset,
                      decorr_strength=DECORR_STRENGTH,
                      noise_fn=NOISE_FN,
                      batch_size=BATCH_SIZE,
                      learning_rate=LEARNING_RATE,
                      lr_milestones=LR_MILESTONES,
                      num_epochs=NUM_EPOCHS,
                      print_epochs=PRINT_EPOCHS)


# In[10]:




# In[11]:


if not utils.is_notebook():
    train_partial = partial(train.train, **kwargs_partial)
   
    MAX_PROCESSES = 2 * torch.cuda.device_count()
    
    if __name__ == "__main__":
        utils.start_timer()
        mp.set_start_method('spawn', force=True)
        
        num_processes = min(len(kwargs_map), MAX_PROCESSES)
        with mp.Pool(num_processes) as p:
            results = utils.kwstarmap(p, train_partial, kwargs_map)

        utils.end_timer_and_print()


# In[37]:


