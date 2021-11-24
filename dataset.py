import os

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
import torchvision.datasets.utils as dataset_utils

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
        elif dataset == 'path' or dataset == 'imagenet':
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
        elif dataset == 'imagenet' or dataset == 'path':
            self.data = ImageFolder(root=root, transform=transform)
        else:
            raise ValueError(dataset)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]

        return x, y, idx
            
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        j = np.random.randint(0, len(self.datasets[1]), size=1)
        j = j[0]

        return tuple([self.datasets[0][i], self.datasets[1][j]])
        # return tuple(d[i] for d in self.datasets)

    def __len__(self):
        all_len = torch.Tensor([len(d) for d in self.datasets])
        # Check datasets are of equal length
        assert (all_len == all_len[0]).all()

        return min(len(d) for d in self.datasets)

def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr


class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
        train1_set.append((Image.fromarray(colored_arr), binary_label))
      elif idx < 40000:
        train2_set.append((Image.fromarray(colored_arr), binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr), binary_label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    dataset_utils.makedir_exist_ok(colored_mnist_dir)
    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))