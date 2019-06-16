import os
import sys
from PIL import Image

import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(".."))
from util.cutout import Cutout


class CustomCIFAR10(datasets.CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)
        self.hdict = None
        self.jitter_transform = transforms.Compose([])
        self.cut_transform = Cutout(n_holes=-1, length=-1)
        self.num_processed = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.set_jitters()
        self.set_cutout()
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.train:
            img = self.jitter_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            img = self.cut_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        self.num_processed += 1
        return img, target

    def set_hparams(self, hparam_tensor, hdict):
        self.num_processed = 0
        self.hparam_tensor = hparam_tensor
        self.hdict = hdict
        self.set_jitters()
        self.set_cutout()

    def reset_hparams(self):
        self.hdict = None

    def set_jitters(self):
        if self.hdict == None:
            return

        eps = 1e-3
        jitter_names = ['hue', 'contrast', 'bright', 'sat']
        jitter_dict = {jname:0 for jname in jitter_names} # Default jitter value is 0 (no jitter).
        for jname in [jname for jname in jitter_names if jname in self.hdict]:
            jitter_idx = self.hdict[jname].index
            jitter = self.hparam_tensor[self.num_processed,jitter_idx].item()
            jitter = (1-2*eps) * jitter + eps
            jitter_dict[jname] = jitter
        
        self.jitter_transform = transforms.ColorJitter(jitter_dict['bright'], 
            jitter_dict['contrast'], jitter_dict['sat'], jitter_dict['hue'])

    def set_cutout(self):
        # Set default values for cutout length, cutout no. of holes.
        if self.hdict == None:
            return
        
        if 'cutlength' or 'cutholes' in self.hdict:
            self.cut_transform.length = 8
            self.cut_transform.n_holes = 1   
        if 'cutlength' in self.hdict:
            cutl_idx = self.hdict['cutlength'].index
            length = self.hparam_tensor[self.num_processed, cutl_idx].item()
            self.cut_transform.length = int(length)
        if 'cutholes' in self.hdict:
            cuth_idx = self.hdict['cutholes'].index
            holes = self.hparam_tensor[self.num_processed, cuth_idx].item()
            self.cut_transform.n_holes = int(holes)