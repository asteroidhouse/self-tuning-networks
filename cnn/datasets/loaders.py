import os
import sys
sys.path.insert(0, 'datasets')

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from util.cutout import Cutout
from cifar import CustomCIFAR10


def create_loaders(args, hyper=False, root_dir='data/'):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])

    if not hyper:
        train_transform.transforms.append(transforms.ColorJitter(hue=args.hue,
            contrast=args.contrast, saturation=args.saturation,
            brightness=args.brightness))

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if not hyper:
        train_transform.transforms.append(Cutout(n_holes=args.cutholes, length=args.cutlength))

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if hyper:
        # Train set
        trainset = CustomCIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        num_train = int(np.floor((1-args.percent_valid) * len(trainset)))

        trainset.train_data = trainset.train_data[:num_train, :, :, :]
        trainset.train_labels = trainset.train_labels[:num_train]
        
        # Validation set
        valset = CustomCIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        valset.train_data = valset.train_data[num_train:, :, :, :]
        valset.train_labels = valset.train_labels[num_train:]
        
        # Test set
        testset = CustomCIFAR10(root=root_dir, train=False, download=True, transform=test_transform)

    else:
        # Train set
        trainset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        num_train = int(np.floor((1-args.percent_valid) * len(trainset)))

        trainset.train_data = trainset.train_data[:num_train, :, :, :]
        trainset.train_labels = trainset.train_labels[:num_train]
        # Validation set
        valset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        valset.train_data = valset.train_data[num_train:, :, :, :]
        valset.train_labels = valset.train_labels[num_train:]
        # Test set
        testset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=False)
    valid_loader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=True, pin_memory=False)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, pin_memory=False)

    return train_loader, valid_loader, test_loader
