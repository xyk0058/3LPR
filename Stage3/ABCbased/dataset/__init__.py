# from dataset.cifar import CIFAR10
from mypath import Path
from dataset.imbSmallImageNet127 import ImbSmallImageNet127


# def cifar10(regim='train', root=Path.db_root_dir('cifar10'), transform=None):
#     return CIFAR10(root=root, train=regim=='train', download=False, transform=transform)


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
