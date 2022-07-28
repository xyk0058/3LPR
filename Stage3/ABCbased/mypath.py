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
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
