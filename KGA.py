import Preprocessing 
import Pretraining

import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import itertools


import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18
import torch.nn.functional as F


import argparse
import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args, frozen=False, layers=[]):

    RNG = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # Initializing the data
    
    if args.model == 'Cifar10':
        num_classes = 10
        _, _, retain_set, forget_set, unseen_set = Preprocessing.load_CIFAR10(args)
    elif args.model == 'Cifar100':
        num_classes =100
        _, _, retain_set, forget_set, unseen_set = Preprocessing.load_CIFAR100(args)

    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
    )

    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
    )
    unseen_loader = torch.utils.data.DataLoader(
        unseen_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
    )



    # Loading in the pre-trained models
    path = f'./Pre-trained models/{args.archetype}/{args.model}'
    if args.archetype == 'Resnet18':
        model_ad = torchvision.models.resnet18(weights=None, num_classes=num_classes).to(DEVICE)
        model_af = torchvision.models.resnet18(weights=None, num_classes=num_classes).to(DEVICE)
        model_an = torchvision.models.resnet18(weights=None, num_classes=num_classes).to(DEVICE)
        model_astar = torchvision.models.resnet18(weights=None, num_classes=num_classes).to(DEVICE)
    elif args.archetype == 'VGG11':
        model_ad = torchvision.models.vgg11(weights=None, num_classes=num_classes).to(DEVICE)
        model_af = torchvision.models.vgg11(weights=None, num_classes=num_classes).to(DEVICE)
        model_an = torchvision.models.vgg11(weights=None, num_classes=num_classes).to(DEVICE)
        model_astar = torchvision.models.vgg11(weights=None, num_classes=num_classes).to(DEVICE)

    model_ad.load_state_dict(torch.load(os.path.join(path, f"Model_Ad_{args.seed}.pt"))['model_state_dict'])
    model_af.load_state_dict(torch.load(os.path.join(path, f"Model_Af_{args.seed}.pt"))['model_state_dict'])
    model_an.load_state_dict(torch.load(os.path.join(path, f"Model_An_{args.seed}.pt"))['model_state_dict'])
    model_astar.load_state_dict(torch.load(os.path.join(path, f"Model_Ad_{args.seed}.pt"))['model_state_dict'])    


    #Setting the parameters to be trained
    for model1, model2, model3 in zip(model_ad.parameters(), model_af.parameters(), model_an.parameters()):
        model1.requires_grad = False
        model2.requires_grad = False
        model3.requires_grad = False

    if frozen:
        with torch.no_grad():
            for n, p in model_astar.named_parameters():
                for layer in layers:
                    if args.archetype == 'Resnet18':
                        if n.startswith(f'layer{layer}'):
                            p.requires_grad = False
                    if args.archetype == 'VGG11':
                        if n.startswith(f'features{layer}'):
                            p.requires_grad = False
            

    # Initializing the KGA model
    model_astar.train()
    optimizer = torch.optim.Adam(model_astar.parameters(), lr=args.lr, eps=args.adam_epsilon)

    begin = datetime.datetime.now()
    model_to_save = {}
    training_total_loss = []

    # Training sequence for the kga 
    for epoch in range(args.max_steps):
        
        start = datetime.datetime.now()
        forget_iterator = iter(forget_loader)
        unseen_iterator = iter(unseen_loader)

        for batch_idx, (inputs, outputs) in  enumerate(retain_loader):
            inputs = inputs.to(DEVICE)
        
            pred_logits = F.log_softmax(model_astar(inputs), dim=-1)
            tgt_logits = F.log_softmax(model_ad(inputs), dim=-1).detach() 
            loss_r  = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='mean')
            
            (loss_r * (1-args.retain_loss_ratio)/args.inner_step).backward()
            
            if batch_idx % args.inner_step == 0:
                try:
                    batch_forget = next(forget_iterator)
                except StopIteration:
                    forget_iterator = iter(forget_loader)
                    batch_forget = next(forget_iterator)
                
                try:
                    batch_unseen = next(unseen_iterator)
                except StopIteration:
                    unseen_iterator = iter(unseen_loader)
                    batch_unseen  = next(unseen_iterator)
                
                batch_forget = batch_forget[0].to(DEVICE)
                batch_unseen = batch_unseen[0].to(DEVICE)

                
                pred_logits = F.log_softmax(model_astar(batch_forget), dim=-1)
                tgt_logits = F.log_softmax(model_af(batch_forget), dim=-1).detach()
                loss_align = F.kl_div(input = pred_logits, target=tgt_logits, log_target=True, reduction='mean')
                
                pred_logits = F.log_softmax(model_an(batch_unseen), dim=-1).detach()
                tgt_logits = F.log_softmax(model_ad(batch_unseen), dim=-1).detach()
                tgt_align = F.kl_div(input = pred_logits, target=tgt_logits, log_target=True, reduction='mean')

                loss_align = torch.abs(loss_align - tgt_align.item())
                loss_align.backward()
                
                total_loss = loss_align.item() + args.alpha * loss_r.item()
                training_total_loss.append(total_loss)

                optimizer.step()
                model_astar.zero_grad()
            
        print(f'Epoch {epoch+1}: \n Loss: {round(total_loss,2)} \n In {round((datetime.datetime.now()-start).total_seconds(),2)} seconds')

    model_to_save['Training time'] = round((datetime.datetime.now() - begin).total_seconds())
    model_to_save['Model_state_dict'] = model_astar.state_dict()
    model_to_save["Training loss"] = training_total_loss

    directory = f'./Unlearned models/{args.archetype}/{args.model}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save({'model_state_dict': model_to_save['Model_state_dict'] ,
                "Loss": model_to_save["Training loss"],
                "Training time": model_to_save["Training time"]},
            os.path.join(directory, f'Model_{args.seed}_{layers[-1]}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', help='Number of epochs', default=15, type=int)
    parser.add_argument('--seed', help='random seed', default=2000, type=int)
    parser.add_argument('--model', help='Cifar10, Cifar100 or googleset', default='Cifar10')
    parser.add_argument('--max_steps', help='number of steps in the outer loop', default=10, type=int)
    parser.add_argument('--inner_step', help='number of steps in the inner loop', default=120)
    parser.add_argument('--batchsize', help="Batchsize for training", default=32, type=int)
    parser.add_argument('--alpha', help='learning rate alpha', default=0.1, type=float)
    parser.add_argument('--lr', help='Learning rate of the model', default=0.05, type=float)
    parser.add_argument('--retain_loss_ratio', help='idk', default=0.1)
    parser.add_argument('--pre_train', help='pre_trains the models', default=False)
    parser.add_argument('--archetype', help='Resent18 or VGG11', default='Resnet18')
    parser.add_argument('--adam_epsilon', help='', default=0.00000001)
    args = parser.parse_args()
    
    # Layer freezing on Resnet18: 1, 2, 3, 4
    # Layer freezing on VGG11: 0, 3, 6, 8

    # # Training for Resnet18, Cifar10
    # main(args, frozen=False, layers=[0])
    # main(args, frozen=True, layers=[1])
    # main(args, frozen=True, layers=[1,2])
    # main(args, frozen=True, layers=[1,2,3])

    # # Training for Resnet18, Cifar100
    # args.model = 'Cifar100'
    # main(args, frozen=False, layers=[0])
    # main(args, frozen=True, layers=[1])
    # main(args, frozen=True, layers=[1,2])
    # main(args, frozen=True, layers=[1,2,3])

    # Training for VGG11, cifar100
    args.archetype = 'VGG11'
    main(args, frozen=False, layers=[0])
    main(args, frozen=True, layers=[0])
    main(args, frozen=True, layers=[0,3])
    main(args, frozen=True, layers=[0,3,6])

    # Training for VGG11, Cifar10
    args.model = 'Cifar10'
    main(args, frozen=False, layers=[0])
    main(args, frozen=True, layers=[0])
    main(args, frozen=True, layers=[0,3])
    main(args, frozen=True, layers=[0,3,6])