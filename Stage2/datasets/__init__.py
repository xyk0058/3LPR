from datasets.cifar import CIFAR10, CIFAR100
from datasets.imblancecifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.miniimagenet import make_dataset, MiniImagenet84
from datasets.imbstl10 import STL10LT
from datasets.imbSmallImageNet127 import ImbSmallImageNet127
from mypath import Path
import os

def cifar10(regim='train', root=Path.db_root_dir('cifar10'), transform=None):
    return CIFAR10(root=root, train=regim=='train', download=False, transform=transform)

def cifar100(regim='train', root=Path.db_root_dir('cifar100'), transform=None):
    return CIFAR100(root=root, train=regim=='train', download=False, transform=transform)

def miniimagenet(root=Path.db_root_dir('miniimagenet'), transform=None, transform_test=None):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = make_dataset(root=root)
    trainset = MiniImagenet84(train_data, train_labels, transform=transform)
    testset = MiniImagenet84(val_data, val_labels, transform=transform_test)
    return trainset, testset

def imblancecifar10(regim='train', root=Path.db_root_dir('cifar10'), transform=None, imb_factor=None):
    print('imblancecifar10')
    return IMBALANCECIFAR10(root=root, train=regim=='train', download=False, transform=transform, imb_factor=imb_factor)

def imblancecifar100(regim='train', root=Path.db_root_dir('cifar100'), transform=None, imb_factor=None):
    print('imblancecifar100')
    return IMBALANCECIFAR100(root=root, train=regim=='train', download=False, transform=transform, imb_factor=imb_factor)

def imblancestl10(regim='train', root=Path.db_root_dir('stl10'), transform=None, imb_factor=None):
    return STL10LT(root=root, train=regim=='train', download=False, transform=transform, imb_factor=imb_factor)

def smallimagenet127_x32(regim='train', root=None, transform=None):
    if regim == 'train':
        root = Path.db_root_dir('smallimagenet127_x32_train')
        train = True
    else:
        root = Path.db_root_dir('smallimagenet127_x32_val')
        train = False
    return ImbSmallImageNet127(root=root, transform=transform, train=train)

def smallimagenet127_x64(regim='train', root=None, transform=None):
    if regim == 'train':
        root = Path.db_root_dir('smallimagenet127_x64_train')
        train = True
    else:
        root = Path.db_root_dir('smallimagenet127_x64_val')
        train = False
    return ImbSmallImageNet127(root=root, transform=transform, train=train)
