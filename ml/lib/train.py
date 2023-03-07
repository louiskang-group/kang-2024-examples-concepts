"""

train.py:   Functions for training neural networks on single tasks and two tasks.

"""

__author__ = "Louis Kang"
__date__ = "2023/03/07"
__license__ = "GPLv3"
__reference__ = "To be determined"



import os, sys
import gc

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import utils
import ml


# Train single-task networks
def train(device, model, mode,
          train_dataset, test_dataset,
          decorr_criterion, decorr_strength,
          noise_fn, batch_size, learning_rate,
          num_epochs, print_epochs):
    
    torch.seed()
    
    # Allow for defining models as a tuple as (model_name, model_keyword_arguments).
    # This is helpful for multiprocessing because the models can be created in each
    # thread instead of being all created by the parent thread and sent to each child.
    if isinstance(model, tuple):
        model = model[0](**model[1]).to(device)
    else:
        model = model.to(device)
    
    # Similarly, create datasets locally in each child thread.
    set_seed = True
    if isinstance(train_dataset, tuple):
        train_dataset = train_dataset[0](**train_dataset[1])
        set_seed = False
    if isinstance(test_dataset, tuple):
        test_dataset = test_dataset[0](**test_dataset[1])
        set_seed = False
    train_size = len(train_dataset)
    num_batches = train_size / batch_size
    if num_batches.is_integer():
        num_batches = int(num_batches)
    else:
        raise Exception(f"num_batches must divide {train_size},"
                        " the size of train_dataset")
   
    # For MNIST, GPU memory generally large enough to load entire train subset
    # Set identification
    if mode == 'set':
        train_X, _, train_target = next(iter(
            DataLoader(train_dataset, batch_size=train_size)
        ))
    # Digit classification
    elif mode in {'testval', 'trainval'}:
        train_X, train_target = next(iter(
            DataLoader(train_dataset, batch_size=train_size)
        ))
    else:
        raise Exception("mode must be 'testval', 'trainval', or 'set'")
    train_X = train_X.to(device)
    train_target = train_target.to(device)
    
    X_batch_dims = (num_batches, batch_size, *train_X.shape[1:])
    target_batch_dims = (num_batches, batch_size, *train_target.shape[1:])
    
    if set_seed:
        rand_seed = torch.sum(
            train_target * torch.arange(len(train_target), device=device)
        ).item()
        torch.manual_seed(rand_seed)
        
    # Validate with held-out test dataset
    if mode == 'testval':
        # For MNIST, GPU memory generally large enough to load entire test dataset
        test_size = len(test_dataset)
        test_X, test_target = next(iter(
            DataLoader(test_dataset, batch_size=test_size)
        ))
        test_X = test_X.to(device)
        test_target = test_target.to(device)
    # Validate with noisy train dataset
    elif mode in {'trainval', 'set'}:
        test_X = noise_fn(train_X)
        test_target = train_target
        
    
    class_criterion = nn.CrossEntropyLoss(reduction='sum')
    set_criterion = nn.CrossEntropyLoss(reduction='sum')
    if decorr_criterion is None:
        decorr_strength = 0.
        # This allows for evaluation of correlations even in the baseline condition
        decorr_criterion = ml.decorr_criterion
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    loss_result = []
    train_accuracy_result = []
    test_accuracy_result = []

    
    for t in range(num_epochs):
        
        losses = np.array([0., 0.])
        train_correct = 0
        
        model.train()
        # Generate randomly ordered batches
        train_perm = torch.randperm(train_size)
        X_batches = train_X[train_perm].view(X_batch_dims)
        target_batches = train_target[train_perm].view(target_batch_dims)
        
        for X, target in zip(X_batches, target_batches):
            activations, logits = model(X)
            target_loss = class_criterion(logits, target)
            decorr_loss = decorr_criterion(activations)
            loss = target_loss + decorr_strength * decorr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                losses += np.array([target_loss.item(), decorr_loss.item()])
                train_correct += utils.logits_to_accuracy(logits, target,
                                                          reduction='sum')
        
        losses /= train_size
        loss_result.append(losses)
        train_accuracy = train_correct / train_size
        train_accuracy_result.append(train_accuracy)
        
        # During each epoch, validate trained network on full test dataset
        model.eval()
        with torch.no_grad():
            _, logits = model(test_X)
            test_accuracy = utils.logits_to_accuracy(logits, test_target,
                                                     reduction='mean')
            test_accuracy_result.append(test_accuracy)
                
        if (print_epochs > 0
            and ((t+1) % print_epochs == 0 or t == 0)):    
            print(f"{device} {t:>3d})"
                  f"   target: {losses[0]:>6.3f}  decorr: {losses[1]:>6.3f}"
                  f"   train: {train_accuracy:>5.3f}  test: {test_accuracy:>5.3f}")
        
    loss_result = np.array(loss_result)
    train_accuracy_result = np.array(train_accuracy_result)
    test_accuracy_result = np.array(test_accuracy_result)

    # Save activations of trained network with test dataset
    model.eval()
    with torch.no_grad():
        activations, _ = model(test_X)
    activations = activations.cpu().numpy()
        
    # Free GPU memory
    del train_X, train_target, X_batches, target_batches
    del test_X, test_target, model
    gc.collect()
    torch.cuda.empty_cache()
    
    return (loss_result,
            train_accuracy_result, test_accuracy_result,
            activations)



# See comments above for the train function for more information
def two_target_train(device, model, mode,
                     train_dataset, test_dataset,
                     decorr_criterion, decorr_strength,
                     noise_fn, batch_size, learning_rate,
                     num_epochs, print_epochs):
    
    torch.seed()
    
    if isinstance(model, tuple):
        model = model[0](**model[1]).to(device)
    else:
        model = model.to(device)
        
    set_seed = True
    if isinstance(train_dataset, tuple):
        train_dataset = train_dataset[0](**train_dataset[1])
        set_seed = False
    if isinstance(test_dataset, tuple):
        test_dataset = test_dataset[0](**test_dataset[1])
        set_seed = False
    train_size = len(train_dataset)
    num_batches = train_size / batch_size
    if num_batches.is_integer():
        num_batches = int(num_batches)
    else:
        raise Exception(f"num_batches must divide {train_size},"
                        " the size of train_dataset")
    
    # y is class target, z is set target
    train_X, train_y, train_z = next(iter(
        DataLoader(train_dataset, batch_size=train_size)
    ))
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    train_z = train_z.to(device)
    print_X = torch.mean(train_X[0]).item()
    print_z = train_z[0].item()
    
    X_batch_dims = (num_batches, batch_size, *train_X.shape[1:])
    y_batch_dims = (num_batches, batch_size, *train_y.shape[1:])
    z_batch_dims = (num_batches, batch_size, *train_z.shape[1:])
    
    if set_seed:
        rand_seed = torch.sum(
            train_z * torch.arange(len(train_z), device=device)
        ).item()
        torch.manual_seed(rand_seed)
    
    # trainsave and testsave save activations and logits of trained network with
    # train and test datasets, respectively. The former is the default option.
    if isinstance(mode, str):
        mode = [mode, 'trainsave']
    if mode[0] != 'trainval' and mode[0] != 'testval':
        raise Exception("mode[0] must be 'trainval' or 'testval'")
    if mode[1] != 'trainsave' and mode[1] != 'testsave':
        raise Exception("mode[1] must be 'trainsave' or 'testsave'")
    
    if mode[0] == 'trainval':
        test_X = noise_fn(train_X)
        test_y = train_y
        test_z = train_z
    # Digit classification and set identification are validated with different
    # test datasets.
    elif mode[0] == 'testval':
        test_size = len(test_dataset)
        test_X_for_y, test_y = next(iter(
            DataLoader(test_dataset, batch_size=test_size)
        ))
        test_X_for_y = test_X_for_y.to(device)
        test_y = test_y.to(device)
        test_X_for_z = noise_fn(train_X)
        test_z = train_z
    
    class_criterion = nn.CrossEntropyLoss(reduction='sum')
    set_criterion = nn.CrossEntropyLoss(reduction='sum')
    if decorr_criterion is None:
        decorr_strength = 0.
        decorr_criterion = ml.decorr_criterion
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    loss_result = []
    train_accuracy_result = []
    test_accuracy_result = []
    activation_result = []
    

    for t in range(num_epochs):
        
        # Digit classification loss, set identification loss, DeCorr/HalfCorr loss
        losses = np.array([0., 0., 0.])
        # Digit classification accuracy, set identification accuracy
        train_correct = np.array([0, 0])
        
        model.train()
        train_perm = torch.randperm(train_size)
        X_batches = train_X[train_perm].view(X_batch_dims)
        y_batches = train_y[train_perm].view(y_batch_dims)
        z_batches = train_z[train_perm].view(z_batch_dims)
        
        for X, y, z in zip(X_batches, y_batches, z_batches):
            activations, y_logits, z_logits = model(X)
            class_loss = class_criterion(y_logits, y)
            set_loss = set_criterion(z_logits, z)
            decorr_loss = decorr_criterion(activations)
            loss = (class_loss + set_loss + decorr_strength * decorr_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                losses += np.array([class_loss.item(), set_loss.item(),
                                    decorr_loss.item()])
                train_correct += np.array([
                    utils.logits_to_accuracy(y_logits, y, reduction='sum'),
                    utils.logits_to_accuracy(z_logits, z, reduction='sum')
                ])
        
        losses /= train_size
        loss_result.append(losses)
        train_accuracies = train_correct / train_size
        train_accuracy_result.append(train_accuracies)
        
        model.eval()
        with torch.no_grad():
            if mode[0] == 'trainval':
                _, y_logits, z_logits = model(test_X)
            elif mode[0] == 'testval':
                _, y_logits, _ = model(test_X_for_y)
                _, _, z_logits = model(test_X_for_z)
                
            test_accuracies = np.array([
                utils.logits_to_accuracy(y_logits, test_y, reduction='mean'),
                utils.logits_to_accuracy(z_logits, test_z, reduction='mean')
            ])
            test_accuracy_result.append(test_accuracies)
                
        if (print_epochs > 0
            and ((t+1) % print_epochs == 0 or t == 0)):    
            print(f"{device} {t:>3d})"
                  f"   loss: {losses[0]:>6.3f} {losses[1]:>6.3f}  {losses[2]:>.3e}"
                  f"   train: {train_accuracies[0]:>5.3f} {train_accuracies[1]:>5.3f}"
                  f"   test: {test_accuracies[0]:>5.3f} {test_accuracies[1]:>5.3f}")
        
    loss_result = np.array(loss_result)
    train_accuracy_result = np.array(train_accuracy_result)
    test_accuracy_result = np.array(test_accuracy_result)
        
    model.eval()
    weights = (model.classifier1.weight.detach().cpu().numpy(),
               model.classifier2.weight.detach().cpu().numpy())
    biases = (model.classifier1.bias.detach().cpu().numpy(),
              model.classifier2.bias.detach().cpu().numpy())
    # Save activations with train dataset
    if mode[1] == 'trainsave':
        with torch.no_grad():
            activations, _, _ = model(train_X)
        activations = activations.cpu().numpy()
        targets = (train_y.cpu().numpy(), train_z.cpu().numpy())
    # Save activations with test datasets
    elif mode[1] == 'testsave':
        if mode[0] == 'trainval':
            with torch.no_grad():
                activations, _, _ = model(test_X)
            activations = activations.cpu().numpy()
        elif mode[0] == 'testval':
            with torch.no_grad():
                y_activations, _, _ = model(test_X_for_y)
                z_activations, _, _ = model(test_X_for_z)
            activations = (y_activations.cpu().numpy(),
                           z_activations.cpu().numpy())
        targets = (test_y.cpu().numpy(), test_z.cpu().numpy())
    
    del train_X, train_y, train_z
    if mode[0] == 'trainval':
        del test_X
    elif mode[0] == 'testval':
        del test_X_for_y, test_X_for_z
        if mode[1] == 'testsave':
            del y_activations, z_activations
    del test_y, test_z, model
    gc.collect()
    torch.cuda.empty_cache()
    
    return (loss_result,
            train_accuracy_result, test_accuracy_result,
            activations, weights, biases, targets)
