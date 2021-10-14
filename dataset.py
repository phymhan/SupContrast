import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100

from util import TwoCropTransform


class ClassDataset(Dataset):
    """Paired ImageFolder (for both ImageNet and StylizedImageNet, in that order)"""
    def __init__(self, dataset, root, transform=None, target_transform=None, is_valid_file=None, is_train=True):
        if dataset == 'cifar10':
            self.data = CIFAR10(root=root,
                                transform=transform,
                                download=True)
        elif dataset == 'cifar100':
            self.data = CIFAR100(root=root,
                                 transform=transform,
                                 download=True)
        elif dataset == 'path':
            self.data = ImageFolder(root=root,
                                    transform=transform)
        else:
            raise ValueError(dataset)

        self.targets = np.array(self.data.targets)

        all_labels = np.unique(self.targets)

        self.label_to_idxs = {int(i): np.where(self.targets==i)[0] for i in all_labels}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, labels, num_imgs=1):
        # Divide total number of images to sample into equal partitions amongst the labels
        parts = [num_imgs // len(labels) + (1 if x < num_imgs % len(labels) else 0)  for x in range(len(labels))]

        x = []
        
        # Sample images from respective classes
        for i in range(len(parts)):
            if parts[i] > 0:
                idxs = np.random.choice(self.label_to_idxs[int(labels[i])], parts[i])
                x.append(torch.cat([torch.stack(self.data[idx][0], dim=0) for idx in idxs], dim=0))
        
        return torch.cat(x, dim=0)

class IdxDataset(Dataset):
    """Paired ImageFolder (for both ImageNet and StylizedImageNet, in that order)"""
    def __init__(self, dataset, root, transform=None, target_transform=None, is_valid_file=None, train=True):
        if dataset == 'cifar10':
            self.data = CIFAR10(root=root,
                                transform=transform,
                                train=train,
                                download=True)
        elif dataset == 'cifar100':
            self.data = CIFAR100(root=root,
                                 transform=transform,
                                 train=train,
                                 download=True)
        elif opt.dataset == 'imagenet':
            self.data = ImageFolder(root=root, transform=TwoCropTransform(transform))
        elif dataset == 'path':
            self.data = ImageFolder(root=root, transform=transform)
        else:
            raise ValueError(dataset)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]

        return x, y, idx
            