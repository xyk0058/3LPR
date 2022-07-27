# class Path(object):
#     @staticmethod
#     def db_root_dir(dataset):
#         if dataset == 'cifar10':
#             return '/ichec/work/ichec007/cifar/'
#         elif dataset == 'cifar100':
#             return '/ichec/work/ichec007/cifar/'
#         elif dataset == 'miniimagenet':
#             return '/ichec/work/ichec007/mini/'
#         else:
#             print('Dataset {} not available.'.format(dataset))
#             raise NotImplementedError

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cifar10':
            return '/home/weilegexiang/Documents/unbalance/datasets/cifar/'
        elif dataset == 'cifar100':
            return '/home/weilegexiang/Documents/unbalance/datasets/cifar/'
        elif dataset == 'miniimagenet':
            return '/home/weilegexiang/Documents/unbalance/datasets/mini/'
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
