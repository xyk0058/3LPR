from pprint import isreadable
import numpy as np
from PIL import Image
from mypath import Path
import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault


# Parameters for data
smallimagenet_mean = (0.485, 0.456, 0.406) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
smallimagenet_std = (0.229, 0.224, 0.225) # equals np.std(train_set.train_data, axis=(0,1,2))/255

# Augmentations.
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(smallimagenet_mean, smallimagenet_std)
    ])

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(smallimagenet_mean, smallimagenet_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(smallimagenet_mean, smallimagenet_std)
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

def get_smallimagenet127(args, root, l_samples, u_samples, transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=False, isrelabel=False):
    train_labeled_idxs, train_unlabeled_idxs = get_split(beta=10)

    train_labeled_dataset = ImbSmallImageNet127(Path.db_root_dir('smallimagenet127_x32_train'), indexs=train_labeled_idxs, train=True, transform=transform_train, isrelabel=isrelabel)
    train_unlabeled_dataset = ImbSmallImageNet127(Path.db_root_dir('smallimagenet127_x32_train'), indexs=train_unlabeled_idxs, train=True,
                                                transform=TransformTwice(transform_train, transform_strong))
    test_dataset = ImbSmallImageNet127(Path.db_root_dir('smallimagenet127_x32_val'), indexs=None, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset,test_dataset, train_labeled_idxs, train_unlabeled_idxs

def get_split(args, beta=10):
    beta = beta * 0.01
    relabel = np.load(args.labels)['arr_0']
    label_idx = np.load(args.labeled)['arr_0']
    print('label, relabel', label_idx.shape, relabel.shape)
    train_labeled_idxs = label_idx
    train_unlabeled_idxs = np.array(list(set(range(1281167))-set(train_labeled_idxs)))
    print('train_unlabeled_idxs', train_labeled_idxs.shape, train_unlabeled_idxs.shape)
    return train_labeled_idxs, train_unlabeled_idxs, relabel


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImbSmallImageNet127(torchvision.datasets.ImageFolder):
    cls_num = 127

    def __init__(self, root, indexs=None, rand_number=0, transform=None, target_transform=None, loader=pil_loader, train=True):
        super(ImbSmallImageNet127, self).__init__(root=root, transform=transform, target_transform=target_transform, loader=loader, isrelabel=False)
        self.root = root
        np.random.seed(rand_number)
        self.train = train
        self.targets = []
        self.img_num_list = [0 for i in range(self.cls_num)]

        if not indexs is None:
            self.samples = self.samples[indexs]

        for i in range(len(self.samples)):
            path, target = self.samples[i]
            self.targets.append(target)
            self.img_num_list[target] = self.img_num_list[target] + 1

        print('ImbSmallImageNet127_img_num_list: ', self.img_num_list)

        
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img2 = self.transform(img)
            img = self.transform(img)

        # print('target1', target, self.train)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target_ = torch.zeros(self.cls_num)
        target_[target] = 1
        
        # print('target2', target)
            
        return {'image':img, 'target':target_, 'index':index, 'image2':img2}
    
    def getAllLabeles(self):
        return self.targets