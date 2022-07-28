import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import math


class STL10LT(torchvision.datasets.STL10):
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.train = train
        if train:
            super(STL10LT, self).__init__(root, split='train',
                 transform=transform, target_transform=target_transform,
                 download=download)

            labeled_data = IMBALANCESTL10(root, imb_type=imb_type, imb_factor=imb_factor, rand_number=rand_number, train=True,
                 transform=transform, target_transform=target_transform,
                 download=download)
            unlabeled_data = STL10_unlabeled(root, indexs=None, split='unlabeled', transform=transform, target_transform=target_transform, download=download)
            print('labeled_data.data', len(labeled_data.data), len(unlabeled_data.data), type(labeled_data.data), type(unlabeled_data.data))
            print('labeled_data.labels', len(labeled_data.labels), len(unlabeled_data.labels))
            self.labeled_data = labeled_data.data
            self.labeled_data_labels = labeled_data.labels

            self.labeled_dataset = labeled_data

            # labeled_data.data.extend(unlabeled_data.data)
            self.data = labeled_data.data
            # labeled_data.labels.extend(unlabeled_data.labels)
            self.targets = labeled_data.labels

            self.img_num_list = labeled_data.img_num_list
            print('stl10 data', len(self.data))
            print('stl10 labels', len(self.targets))
            print('stl10 img_num_list', self.img_num_list)
        else:
            super(STL10LT, self).__init__(root, split='test',
                 transform=transform, target_transform=target_transform,
                 download=download)
            test_dataset = STL10_labeled(root, indexs=None, split='test', transform=transform, download=download)
            self.data = test_dataset.data
            self.targets = test_dataset.labels

    # def __getitem__(self, index):
    #     img, target = self.data[index], self.labels[index].astype(np.int64)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target, index

    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        # print('transform', self.transform)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        # if self.train:
        #     return {'image':self.transform(img), 'image2': self.transform(img), 'target':target, 'index':index}
        
        # return {'image':self.transform(img), 'target': target, 'index':index}
        return {'image':self.transform(img), 'image2': self.transform(img), 'target':target, 'index':index}


    def __len__(self):
        return len(self.data)


class STL10_labeled(torchvision.datasets.STL10):

    def __init__(self, root, indexs=None, split='train',
                 transform=None, target_transform=None,
                 download=False, added_data=None):
        super(STL10_labeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

        if added_data is not None:
            self.data = np.concatenate((self.data, added_data), axis=0)
            self.labels = np.concatenate((self.labels, self.labels[:len(added_data)]), axis=0)

        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)
            

class STL10_unlabeled(torchvision.datasets.STL10):
    def __init__(self, root, indexs=None, split='unlabeled',
                 transform=None, target_transform=None,
                 download=False, added_data=None):
        super(STL10_unlabeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)

        if indexs is not None:
            self.data = self.data[indexs]
            # self.labels = np.array([-1 for i in range(len(self.labels))])
            self.labels = np.array([-1 for i in range(len(self.labels))])

        if added_data is not None:
            self.data = np.concatenate((self.data, added_data), axis=0)
            self.labels = np.concatenate((self.labels, self.labels[:len(added_data)]), axis=0)

        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


class IMBALANCESTL10(torchvision.datasets.STL10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCESTL10, self).__init__(root, split='train',
                 transform=transform, target_transform=target_transform,
                 download=download)
        
        self.train = train
        self.transform = transform

        np.random.seed(rand_number)
        self.select_idx = []
        img_num_list = [len(self.data)/self.cls_num for i in range(self.cls_num)]
        print('img_num_list', img_num_list)
        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            print('dataset [datashape, select_idx]: ', len(self.data), len(self.select_idx))
            imb_factor_str = int(math.ceil(1. / imb_factor))
            np.savez('select_idx_stl_imb'+str(imb_factor_str)+'.npz', self.select_idx)
        
        self.img_num_list = img_num_list

        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]
    
    def __len__(self):
        return len(self.data)

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
        targets_np = np.array(self.labels, dtype=np.int64)
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
        self.labels = new_targets
        self.select_idx = np.hstack(self.select_idx)
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        
        img, target = self.data[index], self.labels[index]
        
        img = Image.fromarray(img)

        if self.train:
            return {'image1':self.transform(img), 'image2': self.transform(img), 'target':target, 'index':index}
        
        return {'image':self.transform(img), 'target': target, 'index':index}
