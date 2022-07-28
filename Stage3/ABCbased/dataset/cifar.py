import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

# Parameters for data
cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

# Augmentations.
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

def get_cifar10(args, root, l_samples, u_samples, transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=True):
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, relabel, select_idx = train_split(args, base_dataset.targets, l_samples, u_samples)

    train_labeled_dataset = CIFAR10_labeled(args, root, train_labeled_idxs, train=True, transform=transform_train, relabel=relabel)
    train_unlabeled_dataset = CIFAR10_unlabeled(args, root, train_unlabeled_idxs, train=True,
                                                transform=TransformTwice(transform_train, transform_strong))
    test_dataset = CIFAR10_labeled(args, root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset,test_dataset, train_labeled_idxs, train_unlabeled_idxs

def train_split(args, labels, n_labeled_per_class, n_unlabeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    relabel = np.load(args.labels)['arr_0']
    label_idx = np.load(args.labeled)['arr_0']
    select_idx = np.load(args.select_idx)['arr_0']

    print('label_idx_cnt: ', label_idx.shape, max(label_idx), max(select_idx))

    train_labeled_idxs = []
    for i in label_idx:
        train_labeled_idxs.append(select_idx[i])
    train_labeled_idxs = np.array(train_labeled_idxs)

    train_unlabeled_idxs = []
    for idx in select_idx:
        if idx in train_labeled_idxs:
            continue
        train_unlabeled_idxs.append(idx)
    train_unlabeled_idxs = np.array(train_unlabeled_idxs)
    print('train_unlabeled_idxs, train_labeled_idxs: ', train_unlabeled_idxs.shape, train_labeled_idxs.shape)

    return train_labeled_idxs, train_unlabeled_idxs, relabel, select_idx

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, args, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=True, relabel=None):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        guess_label_acc = 0
        if relabel is not None:
            relabel = np.load(args.labels)['arr_0']
            label_idx = np.load(args.labeled)['arr_0']
            select_idx = np.load(args.select_idx)['arr_0']
            for idx in label_idx:
                guess_label = np.argmax(relabel[idx])
                if guess_label == self.targets[select_idx[idx]]:
                    guess_label_acc += 1
                self.targets[select_idx[idx]] = guess_label
            print('guess_label_acc', guess_label_acc, len(label_idx))
        self.idx = np.array(range(len(self.data)))
        self.targets = np.array(self.targets)
        if indexs is not None:
            print('indexs', max(indexs))
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]
            self.idx = self.idx[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

        print('CIFAR10_labeled', len(self.data), train)
        img_per_class = []
        for i in range(10):
            idxs = np.where(self.targets == i)[0]
            img_per_class.append(len(idxs))
        print('img_per_class', img_per_class, self.targets[0])
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.idx[index]
    
    def getAllLabeles(self):
        return self.targets
    

class CIFAR10_unlabeled(torchvision.datasets.CIFAR10):

    def __init__(self, args, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(CIFAR10_unlabeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        #self.targets = np.array([-1 for i in range(len(self.targets))])
        self.idx = np.array(range(len(self.data)))
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.idx = self.idx[indexs]
        self.data = [Image.fromarray(img) for img in self.data]
        print('CIFAR10_unlabeled', len(self.data))
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.idx[index]
    
    
    def getAllLabeles(self):
        return self.targets