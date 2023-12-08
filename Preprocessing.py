import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split

import random
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

import argparse

# Set seed is used for dataset partitioning to ensure reproducible results across runs


# Download and normalize the data
def load_CIFAR10(args):
    torch.manual_seed(args.seed)

    #Normalization of the data
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Downloading the training and unseen sets
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize
    )

    unseen_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=normalize
    )
    
    # Defining the indices of Df and Dn
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path
        )
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]
    
    # Splitting the data into Df and Dn

    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)
    
    # Taking a small subset of the test set a unseen data for training purposes
    
    test_set, unseen_set = random_split(unseen_set, [9000, 1000])
   

    return train_set, test_set, retain_set, forget_set, unseen_set

def load_CIFAR100(args):
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Download CIFAR-100 dataset
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    unseen_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    #Define indices for unlearning 
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path
        )
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]
    
    # Splitting the data into Df and Dn

    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)
    
    # Taking a small subset of the test set a unseen data for training purposes

    test_set, unseen_set = random_split(unseen_set, [9000, 1000])
   

    return train_set, test_set, retain_set, forget_set, unseen_set



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='random seed', default=2000, type=int)
    args = parser.parse_args()
    load_CIFAR10(args)
    load_CIFAR100(args)