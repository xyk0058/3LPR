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
    def __init__(self, root, rand_number=0, transform=None, target_transform=None, loader=pil_loader, train=True):
        super(ImbSmallImageNet127, self).__init__(root=root, transform=transform, target_transform=target_transform, loader=loader)
        self.root = root
        np.random.seed(rand_number)
        self.train = train
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.train:
            return {'image1':self.transform(img), 'image2': self.transform(img), 'target':target, 'index':index}
        
        return {'image':self.transform(img), 'target': target, 'index':index}