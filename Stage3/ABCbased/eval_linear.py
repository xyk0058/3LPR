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
    model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)

    print('dataset', dataset)

    if args.dataset == 'cifar10' or args.dataset == 'imbcifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        nclass = 10
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        nclass =100
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

    #Uncomment the proper lines in the network definition
    mylinear = nn.Linear(128, nclass)
    model.cuda()
    mylinear.cuda()

    epochs = 100
    best = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mylinear.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=0.1)

    for eps in range(epochs):
        model.train()
        mylinear.train()

        tbar = tqdm(train_loader)
        tbar.set_description("Train {}/{}".format(eps, epochs))
        acc = 0
        total = 0
        for i, sample in enumerate(tbar):
            images, target, index = sample["image1"].cuda(), sample["target"].cuda(), sample["index"]
            # forward                                                                                                                                                                                                                                                         
            outputs = model(images)
            outputs = mylinear(outputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            preds = torch.argmax(nn.functional.log_softmax(outputs, dim=1), dim=1)        
            acc += torch.sum(preds == target.data)
            total += preds.size(0)
            
            tbar.set_description("Train {}/{}, loss {:.3f}, lr {:.3f}".format(eps, epochs, loss.item(), optimizer.param_groups[0]['lr']))
        scheduler.step()
        print("Train accuracy {:.4f}".format(100.*acc/total))
        
        model.eval()
        mylinear.eval()
        acc = 0
        total = 0
        tbar = tqdm(test_loader)
        tbar.set_description("Test")
        for i, sample in enumerate(tbar):
            images, target, index = sample["image"].cuda(), sample["target"].cuda(), sample["index"]
            # forward                                                                                                                                                                                                                                                         
            outputs = model(images)
            outputs = mylinear(outputs)
            loss = criterion(outputs, target)
            
            preds = torch.argmax(nn.functional.log_softmax(outputs, dim=1), dim=1)        
            acc += torch.sum(preds == target.data)
            total += preds.size(0)
            
            tbar.set_description("Test loss {:.3f}".format(loss.item()))
        acc = 1.*acc/total
        if acc > best:
            best = acc
        print("Test accuracy {:.4f}, Best {:.4f}".format(acc*100, best*100))



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