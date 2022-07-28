class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cifar10':
            return '/home/weilegexiang/Documents/unbalance/ReLaB/samples/cifar10/'
        elif dataset == 'cifar100':
            return '/home/weilegexiang/Documents/unbalance/ReLaB/samples/cifar100/'
        elif dataset == 'miniimagenet':
            return '/home/weilegexiang/Documents/unbalance/ReLaB/samples/miniImagenet84/'
        elif dataset == 'stl10':
            return '/home/weilegexiang/Documents/unbalance/datasets/stl10/'
        elif dataset == 'smallimagenet127_x32_train':
            return '/home/weilegexiang/980SSD/ImageNetRoot/ILSVRC2012_img_train_x32/'
        elif dataset == 'smallimagenet127_x32_val':
            return '/home/weilegexiang/980SSD/ImageNetRoot/ILSVRC2012_img_val_x32/'
        elif dataset == 'smallimagenet127_x64_train':
            return '/home/weilegexiang/980SSD/ImageNetRoot/ILSVRC2012_img_train_x64/'
        elif dataset == 'smallimagenet127_x64_val':
            return '/home/weilegexiang/980SSD/ImageNetRoot/ILSVRC2012_img_val_x64/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
