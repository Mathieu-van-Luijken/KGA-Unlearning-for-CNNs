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

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score
import json

import argparse
import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def JSD(pred1, pred2, mean_log):
    loss1 = F.kl_div(pred1.log(), mean_log, reduction='batchmean')
    loss2 = F.kl_div(pred2.log(), mean_log, reduction='batchmean')
    return (loss1 + loss2)/2
    

def main(args, frozen=False, layers=[]):
    pre_train_path = f'./Pre-trained models/{args.archetype}/{args.model}'
    model_path = f'./Unlearned models/{args.archetype}/{args.model}'

    if args.model == 'Cifar10':
        num_classes = 10
        _, test_set, _, forget_set , _ = Preprocessing.load_CIFAR10(args)
    elif args.model == 'Cifar100':
        num_classes =100
        _, test_set, _, forget_set, _ = Preprocessing.load_CIFAR100(args)
    

    original_model = torchvision.models.resnet18(weights=None, num_classes=num_classes).to(DEVICE)
    original_model.load_state_dict(torch.load(os.path.join(pre_train_path, f"Model_Ad_{args.seed}.pt"))['model_state_dict'])
    
    target_model = torchvision.models.resnet18(weights=None, num_classes=num_classes).to(DEVICE)
    target_model.load_state_dict(torch.load(os.path.join(pre_train_path, f"Model_Ar_{args.seed}.pt"))['model_state_dict'])
    
    unlearn_model = torchvision.models.resnet18(weights=None, num_classes=num_classes).to(DEVICE)
    unlearn_model.load_state_dict(torch.load(os.path.join(model_path, f"Model_{args.seed}_{layers[-1]}.pt"))['model_state_dict'])

    if args.forget_data == False:
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=128, shuffle=False
        )  
    else:
        test_loader = test_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=128, shuffle=False
        )

    with torch.no_grad():
        jsd_original = 0
        jsd_target = 0

        f1_original = 0
        f1_unlearn = 0
        f1_target = 0

        acc_original = 0
        acc_unlearn = 0
        acc_target = 0

        for batch_idx, (input, target) in enumerate(test_loader):

            # loading in the samples
            input, target = input.to(DEVICE), target.to(DEVICE)

            original_outputs = original_model(input)
            unlearned_outputs = unlearn_model(input)
            target_outputs = target_model(input)
            
            # JSD of the original model and unlearn model
            original_output_dist = F.softmax(original_outputs, dim=-1).detach()
            unlearn_output_dist = F.softmax(unlearned_outputs, dim=-1).detach()
            mean_dist = (original_output_dist + unlearn_output_dist)/2

            jsd_original =+ JSD(original_output_dist, unlearn_output_dist, mean_dist).item()

            # JSD of the target model and unlearn model
            target_output_dist = F.softmax(target_outputs, dim=-1).detach()
            unlearn_output_dist = F.softmax(unlearned_outputs, dim=-1).detach()
            mean_dist = (target_output_dist + unlearn_output_dist)/2

            jsd_target =+ JSD(target_output_dist, unlearn_output_dist, mean_dist).item()


            #Compute F1 score
            target = target.cpu().numpy()
            original_outputs = torch.argmax(original_outputs, dim=1).cpu().numpy()
            unlearned_outputs = torch.argmax(unlearned_outputs, dim=1).cpu().numpy()
            target_outputs = torch.argmax(target_outputs, dim=1).cpu().numpy()

            f1_original += f1_score(target, original_outputs, average='macro')
            f1_unlearn += f1_score(target, unlearned_outputs, average='macro')
            f1_target += f1_score(target, target_outputs, average='macro')

            acc_original += accuracy_score(target, original_outputs)
            acc_unlearn += accuracy_score(target, unlearned_outputs)
            acc_target += accuracy_score(target, target_outputs)

        jsd_original = jsd_original/len(test_loader)
        jsd_target = jsd_target/len(test_loader)

        f1_original = f1_original/len(test_loader)
        f1_unlearn = f1_unlearn/len(test_loader)
        f1_target = f1_target/len(test_loader)

        acc_original = acc_original/len(test_loader)
        acc_unlearn = acc_unlearn/len(test_loader)
        acc_target = acc_target/len(test_loader)

    results = {
            'Total_training_time': torch.load(os.path.join(model_path, f"Model_{args.seed}_{layers[-1]}.pt"))['Training time'],
            'JSD_or': jsd_original,
            'JSD_tgt': jsd_target,
            'F1_or': f1_original,
            'F1_un': f1_unlearn,
            'F1_tgt': f1_target,
            "Acc_or": acc_original,
            'Acc_un': acc_unlearn,
            'Acc_tgt': acc_target
            }

    with open(os.path.join(model_path, f'results_{args.seed}_{layers[-1]}_{args.forget_data}.json') ,'w') as json_file:
        json.dump(results, json_file)


            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', help='Number of epochs', default=10, type=int)
    parser.add_argument('--seed', help='random seed', default=2000, type=int)
    parser.add_argument('--model', help='Cifar10, Cifar100 or googleset', default='Cifar10')
    parser.add_argument('--max_steps', help='number of steps in the outer loop', default=1, type=int)
    parser.add_argument('--inner_step', help='', default=10)
    parser.add_argument('--batchsize', help="Batchsize for training", default=256, type=int)
    parser.add_argument('--alpha', help='learning rate alpha', default=0.1, type=float)
    parser.add_argument('--lr', help='Learning rate of the model', default=0.05, type=float)
    parser.add_argument('--retain_loss_ratio', help='idk', default=0.1)
    parser.add_argument('--sigma', help='', default=0.9, type=float)
    parser.add_argument('--archetype', help='Resnet18 or Cifar10', default='Resnet18')
    parser.add_argument('--forget_data', default=True)

    args = parser.parse_args()
    
    main(args, frozen=False, layers=[0])
    main(args, frozen=True, layers=[1])
    main(args, frozen=True, layers=[1,2])
    main(args, frozen=True, layers=[1,2,3])

    # Training for Resnet18, Cifar100
    args.model = 'Cifar100'
    main(args, frozen=False, layers=[0])
    main(args, frozen=True, layers=[1])
    main(args, frozen=True, layers=[1,2])
    main(args, frozen=True, layers=[1,2,3])