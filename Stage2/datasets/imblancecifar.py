import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import math

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.02, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.select_idx = []
        img_num_list = [len(self.data)/self.cls_num for i in range(self.cls_num)]
        print('img_num_list', img_num_list)
        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            print('dataset [datashape, select_idx]: ', self.data.shape, self.select_idx.shape)
            imb_factor_str = int(math.ceil(1. / imb_factor))
            np.savez('select_idx_imb'+str(imb_factor_str)+'_cifar'+str(self.cls_num)+'.npz', self.select_idx)
            np.savez('select_idx.npz', self.select_idx)
            print('imbcifar10 targets', self.targets[0], target_transform)
            # breaks
        self.img_num_list = img_num_list
        self.train = train
        self.transform = transform


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            self.select_idx.append(selec_idx)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        self.select_idx = np.hstack(self.select_idx)
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    # def __getitem__(self, index):
        
    #     img, target = self.data[index], self.targets[index]
        
    #     img = Image.fromarray(img)

    #     if self.train:
    #         return {'image1':self.transform(img), 'image2': self.transform(img), 'target':target, 'index':index}
        
    #     return {'image':self.transform(img), 'target': target, 'index':index}
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img2 = self.transform(img)
            img = self.transform(img)

        # print('target1', target, self.train)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # print('target2', target)
            
        return {'image':img, 'target':target, 'index':index, 'image2':img2}

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

    # def __getitem__(self, index):
        
    #     img, target = self.data[index], self.targets[index]
                
    #     img = Image.fromarray(img)
            
    #     if self.train:
    #         return {'image1':self.transform(img), 'image2': self.transform(img), 'target':target, 'index':index}
        
    #     return {'image':self.transform(img), 'target': target, 'index':index}