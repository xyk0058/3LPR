import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImbSmallImageNet127(torchvision.datasets.ImageFolder):
    cls_num = 127

    def __init__(self, root, rand_number=0, transform=None, target_transform=None, loader=pil_loader, train=True):
        super(ImbSmallImageNet127, self).__init__(root=root, transform=transform, target_transform=target_transform, loader=loader)
        self.root = root
        np.random.seed(rand_number)
        self.train = train
        self.targets = []
        self.img_num_list = [0 for i in range(self.cls_num)]

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