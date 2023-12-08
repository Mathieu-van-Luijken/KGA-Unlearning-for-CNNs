import Preprocessing 
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18, vgg11

import argparse
import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# set random seed is used for dataset partitioning to ensure reproducible results across runs

def pretrain(args):
    RNG = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    

    #Initializing the data
    if args.model == 'Cifar10':
        train_set, _, retain_set, forget_set, unseen_set = Preprocessing.load_CIFAR10(args)
    elif args.model == "Cifar100":
        train_set, _, retain_set, forget_set, unseen_set = Preprocessing.load_CIFAR100(args)

    #Setting the DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
    )

    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
    )

    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=128, shuffle=True, num_workers=2,
    )
    unseen_loader = torch.utils.data.DataLoader(
        unseen_set, batch_size=128, shuffle=False, num_workers=2
    )

    #Initializing and training the models

    if args.model == 'Cifar10':
        num_classes = 10
    elif args.model == 'Cifar100':
        num_classes =100

    if args.archetype == 'Resnet18':
        model_ad = resnet18(weights=None, num_classes=num_classes).to(DEVICE)
        model_af = resnet18(weights=None, num_classes=num_classes).to(DEVICE)
        model_an = resnet18(weights=None, num_classes=num_classes).to(DEVICE)
        model_ar = resnet18(weights=None, num_classes=num_classes).to(DEVICE)
    elif args.archetype == 'VGG11':
        model_ad = vgg11(weights=None, num_classes=num_classes).to(DEVICE)
        model_af = vgg11(weights=None, num_classes=num_classes).to(DEVICE)
        model_an = vgg11(weights=None, num_classes=num_classes).to(DEVICE)
        model_ar = vgg11(weights=None, num_classes=num_classes).to(DEVICE)

    # Setting hyperparameters
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    path = f"./Pre-trained models/{args.archetype}/{args.model}"
    if not os.path.exists(path):
        os.makedirs(path)

    #############
    # Training Ar
    #############
    optimizer = torch.optim.SGD(model_ar.parameters(), lr=lr)
    model_to_save = {}
    training_loss_ar = []
    for epoch in range(args.num_epochs):
        begin = datetime.datetime.now()
        start = datetime.datetime.now()
        running_loss = 0
        for batch_idx, (inputs, targets) in enumerate(retain_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model_ar(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_loss_ar.append(running_loss)
        print(f'Epoch {epoch+1}: \n Loss: {round(running_loss,2)} \n In {round((datetime.datetime.now()-start).total_seconds(),2)} seconds')

    #Saving the relevant data for model Ad
    model_to_save['Training time'] = round((datetime.datetime.now() - begin).total_seconds())
    model_to_save['Model_state_dict'] = model_ar.state_dict()
    model_to_save["Training loss"] = training_loss_ar

    torch.save({'model_state_dict': model_to_save['Model_state_dict'] ,
                "Loss": model_to_save["Training loss"],
                "Training time": model_to_save["Training time"]},
            os.path.join(path, f'Model_Ar_{args.seed}.pt')) 
    
    # #############
    # # Training Ad
    # #############
    # optimizer = torch.optim.SGD(model_ad.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # model_to_save = {}
    # training_loss_ad = []
    # for epoch in range(args.num_epochs):
    #     begin = datetime.datetime.now()
    #     start = datetime.datetime.now()
    #     running_loss = 0
    #     for batch_idx, (inputs, targets) in enumerate(train_loader):
    #         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = model_ad(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     training_loss_ad.append(running_loss)
    #     print(f'Epoch {epoch+1}: \n Loss: {round(running_loss,2)} \n In {round((datetime.datetime.now()-start).total_seconds(),2)} seconds')

    # #Saving the relevant data for model Ad
    # model_to_save['Training time'] = round((datetime.datetime.now() - begin).total_seconds())
    # model_to_save['Model_state_dict'] = model_ad.state_dict()
    # model_to_save["Training loss"] = training_loss_ad

    # torch.save({'model_state_dict': model_to_save['Model_state_dict'] ,
    #             "Loss": model_to_save["Training loss"],
    #             "Training time": model_to_save["Training time"]},
    #         os.path.join(path, f'Model_Ad_{args.seed}.pt')) 

    # #############    
    # # Training Af
    # #############
    # optimizer = torch.optim.SGD(model_af.parameters(), lr=lr)
    # model_to_save = {}
    # training_loss_af = []
    # for epoch in range(args.num_epochs):
    #     begin = datetime.datetime.now()
    #     start = datetime.datetime.now()
    #     running_loss = 0
    #     for batch_idx, (inputs, targets) in enumerate(forget_loader):
    #         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = model_af(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     training_loss_af.append(running_loss)
    #     print(f'Epoch {epoch+1}: \n Loss: {round(running_loss,2)} \n In {round((datetime.datetime.now()-start).total_seconds(),2)} seconds')

    # #Saving the relevant data for model Ad
    # model_to_save['Training time'] = round((datetime.datetime.now() - begin).total_seconds())
    # model_to_save['Model_state_dict'] = model_af.state_dict()
    # model_to_save["Training loss"] = training_loss_af

    # torch.save({'model_state_dict': model_to_save['Model_state_dict'] ,
    #             "Loss": model_to_save["Training loss"],
    #             "Training time": model_to_save["Training time"]},
    #         os.path.join(path, f'Model_Af_{args.seed}.pt')) 
        
    # #############
    # # Training An
    # #############
    # optimizer = torch.optim.SGD(model_an.parameters(), lr=lr)
    # model_to_save = {}
    # training_loss_an = []
    # for epoch in range(args.num_epochs):
    #     begin = datetime.datetime.now()
    #     start = datetime.datetime.now()
    #     running_loss = 0
    #     for batch_idx, (inputs, targets) in enumerate(unseen_loader):
    #         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = model_an(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     training_loss_an.append(running_loss)
    #     print(f'Epoch {epoch+1}: \n Loss: {round(running_loss,2)} \n In {round((datetime.datetime.now()-start).total_seconds(),2)} seconds')

    # #Saving the relevant data for model Ad
    # model_to_save['Training time'] = round((datetime.datetime.now() - begin).total_seconds())
    # model_to_save['Model_state_dict'] = model_an.state_dict()
    # model_to_save["Training loss"] = training_loss_an

    # torch.save({'model_state_dict': model_to_save['Model_state_dict'] ,
    #             "Loss": model_to_save["Training loss"],
    #             "Training time": model_to_save["Training time"]},
    #         os.path.join(path, f'Model_An_{args.seed}.pt')) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', help='Number of epochs', default=20, type=int)
    parser.add_argument('--seed', help='random seed', default=2000, type=int)
    parser.add_argument('--model', help='Cifar10, Cifar100 or googleset', default='Cifar10')
    parser.add_argument('--archetype', help='Resnet18 or VGG11', default='Resnet18')
    args = parser.parse_args()
    
    # Train with Cifar10 and Resnet18
    pretrain(args)
    
    # Pretrain with Cifar 100 and Resnet18
    args.model = 'Cifar100'
    pretrain(args)

    # Pretrain with Cifar 100 and VGG11
    args.archetype = 'VGG11'
    pretrain(args)

    # Pretrain with Cifar10 and VGG11
    args.model = 'Cifar10'
    pretrain(args)


