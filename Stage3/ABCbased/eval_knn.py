# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
from __future__ import print_function
import argparse
import os
import shutil
import time
import random
import math
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import wideresnetwithABC as models
import matplotlib.pyplot as plt
from tqdm import tqdm
import dataset as datasets
from sklearn import manifold
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter
from scipy import optimize
parser = argparse.ArgumentParser(description='PyTorch ReMixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result', help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# Method options
parser.add_argument('--num_max', type=int, default=1000, help='Number of samples in the maximal class')
parser.add_argument('--label_ratio', type=float, default=20, help='percentage of labeled data')
parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio')
parser.add_argument('--val-iteration', type=int, default=500, help='Frequency for the evaluation')

# Hyperparameters for ReMixMatch
parser.add_argument('--mix_alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=1.5, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--w_rot', default=0.5, type=float)
parser.add_argument('--w_ent', default=0.5, type=float)
parser.add_argument('--align', action='store_false', help='Distribution alignment term')
#dataset and imbalanced type
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--imbalancetype', type=str, default='long', help='Long tailed or step imbalanced')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
if args.dataset=='cifar10':
    import dataset.remix_cifar10 as dataset
    print(f'==> Preparing imbalanced CIFAR10')
    num_class = 10
elif args.dataset=='svhn':
    import dataset.remix_svhn as dataset
    print(f'==> Preparing imbalanced SVHN')
    num_class = 10
elif args.dataset=='cifar100':
    import dataset.remix_cifar100 as dataset
    print(f'==> Preparing imbalanced CIFAR100')
    num_class = 100
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
# np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio,args.imbalancetype)
    if args.num_max == 5000:
        U_SAMPLES_PER_CLASS = make_imb_data(1, num_class, 1, args.imbalancetype)
    else:
        U_SAMPLES_PER_CLASS = make_imb_data((100-args.label_ratio)/args.label_ratio * args.num_max, num_class, args.imb_ratio,args.imbalancetype)
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)
    ir2=N_SAMPLES_PER_CLASS[-1]/np.array(N_SAMPLES_PER_CLASS)
    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set,test_set, _, _ = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
    elif args.dataset == 'svhn':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_SVHN('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
    elif args.dataset =='cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                            drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("==> creating WRN-28-2 with ABC")

    def create_model(ema=False):
        model = models.WideResNet(num_classes=num_class)
        model = model.cuda()

        params = list(model.parameters())
        if ema:
            for param in params:
                param.detach_()

        return model, params

    model, params = create_model()
    ema_model,  _ = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))


    # Resume
    title = 'ABCremix-'+args.dataset
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.out = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    print('checkpoint state_dict', checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)

    print('dataset', dataset)

    if args.dataset == 'cifar10' or args.dataset == 'imbcifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'miniimagenet':
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]

    size = 32

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    if args.dataset == "cifar10" or args.dataset == 'imbcifar10':
        # trainset = datasets.cifar10(transform=transform, regim='train')
        trainset = datasets.cifar10(transform=transform, regim='train')
        testset = datasets.cifar10(transform=transform, regim='val')
    elif args.dataset == "cifar100":
        trainset = datasets.cifar100(transform=transform, regim='train')
        testset = datasets.cifar100(transform=transform, regim='val')
    elif args.dataset == "miniimagenet":
        trainset, testset = datasets.miniimagenet(transform=transform, transform_test=transform)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    # tracking variables
    total = 0
    train_feats = torch.zeros((len(train_loader.dataset), 128))

    with torch.no_grad():
        tbar = tqdm(train_loader)
        tbar.set_description("Computing features on the train set")
        for i, sample in enumerate(tbar):
            images, index = sample["image1"].cuda(), sample["index"]
            # forward
            features = model(images)
            train_feats[index] = features.detach().cpu()


    trainFeatures = train_feats.cuda()
    trainLabels = torch.LongTensor(train_loader.dataset.targets).cuda()

    trainFeatures = trainFeatures.t()
    C = trainLabels.max() + 1
    C = C.item()

    # start to evaluate

    top1 = 0.
    top5 = 0.
    tbar = tqdm(test_loader)
    tbar.set_description("kNN eval")
    K = 200
    sigma = 0.1
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for i, sample in enumerate(tbar):
            images, targets = sample["image"].cuda(), sample["target"].cuda()
            batchSize = images.size(0)
            # forward                                                                                                                                                                                                                                                         
            features = model(images)
            
            # cosine similarity                                                                                                                                                                                                                                               
            dist = torch.mm(features, trainFeatures)
            
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C),
                                        yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)
            
            # Find which predictions match the target                                                                                                                                                                                                                        
            correct = predictions.eq(targets.data.view(-1,1))
            
            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()
            
            total += targets.size(0)

        print("kNN accuracy", top1/total*100)
        print(top1, total)


def make_imb_data(max_num, class_num, gamma,imb):
    if imb == 'long':
        mu = np.power(1/gamma, 1/(class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    if imb=='step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)
    return list(class_num_list)

if __name__ == '__main__':
    main()