import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from utils import multi_class_loss
from tqdm import tqdm
import math

class GPR_prop(torch.nn.Module):
    '''
    propagation class for GPR_GNN
    '''
    def __init__(self, K, alpha=0.1):
        super(GPR_prop, self).__init__()
        self.K = K
        self.alpha = alpha
        bound = np.sqrt(3/(K+1))
        TEMP = np.random.uniform(-bound, bound, K+1)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        self.temp = nn.Parameter(torch.FloatTensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K
    
    def forward(self, x, A_hat):
        num_nodes, d_model = x.size()
        output = self.temp[0] * x
        for i in range(self.K):
            x = A_hat @ x
            output = output + self.temp[i+1] * x
        return output


class GPRGNN(torch.nn.Module):
    def __init__(self, K, input_dim, d_model, output_dim, nodes_num):
        super(GPRGNN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, d_model, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(d_model, output_dim)
        )
        self.prop = GPR_prop(K)
        self.d_model = d_model
        self.nodes_num = nodes_num
        self.prop.reset_parameters()
        self.linear2 = nn.Sequential(
            nn.Linear(d_model, output_dim, bias=True),
        )
        # self.relu = nn.LeakyReLU(0.2)
        # self.E = torch.nn.Parameter(torch.zeros(nodes_num, d_model))
        # nn.init.xavier_uniform_(self.E.data, gain=np.sqrt(2.0))
    
    def forward(self, x):
        A_hat = torch.softmax(x @ x.T, dim=-1) + torch.eye(x.shape[0], device=x.device)
        x = self.linear(x)
        out = self.prop(x, A_hat)
        return out


class Trainer_K(object):
    def __init__(self, X, labels, labeled_idx, classes, nodes_num):
        self.X = torch.from_numpy(X).cuda()
        # Wn = (self.X @ self.X.T + torch.eye(nodes_num, device=self.X.device)).detach()
        # print("torch", Wn.shape)
        # model = GPRGNN(K=10, input_dim=128, d_model=256, output_dim=classes, nodes_num=nodes_num, Wn=Wn)
        print('AGPGNN_K classes', classes)
        model = GPRGNN(K=10, input_dim=128, d_model=256, output_dim=classes, nodes_num=nodes_num)
        self.model = nn.DataParallel(model).cuda()
        self.split_cnt = 50000
        # self.model = model.cuda()
        self.labels = torch.from_numpy(labels).long().cuda()
        self.labeled_idx = torch.from_numpy(labeled_idx).cuda()
        self.unlabeled_idx = torch.LongTensor(list(set(range(labels.shape[0]))-set(labeled_idx))).cuda()
        print('X', self.X.shape, type(self.X))
        print('labels', self.labels.shape, type(self.labels), max(labels), min(labels))
        print('labeled_idx', self.labeled_idx.shape, type(self.labeled_idx))
        print('unlabeled_idx', self.unlabeled_idx.shape, type(self.unlabeled_idx))
        print('nodes_num', nodes_num)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        # self.targets = torch.zeros((X.shape[0], classes)).cuda().scatter_(1, self.labels.view(-1, 1), 1)
        self.targets = torch.zeros((X.shape[0], classes)).cuda()
        # for c in range(classes):
        #     cur_idx = labeled_idx[labels[labeled_idx] == c]
        #     self.targets[:,c] = 1 / cur_idx.shape[0]
        #     print('cur_idx', cur_idx.shape[0])
        for idx in labeled_idx:
            c = self.labels[idx]
            self.targets[idx] = 0
            self.targets[idx][c] = 1
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        self.best = 0
        self.best_epoch = 0
        self.acc = []
        self.classes = classes
        
    def _train_subgraph(self, X, labeled_idx, unlabeled_idx, labels, targets):
        acc, total = 0, 0
        
        self.optimizer.zero_grad()
        outputs = self.model(X)

        emb = outputs.clone()
        
        outputs = outputs[labeled_idx]
        
        target = targets[labeled_idx]
        
        outputs_l = emb[labeled_idx]
        outputs_u = emb[unlabeled_idx]
        ul = torch.mean(outputs_l, dim=0)
        uu = torch.mean(outputs_u, dim=0)
        ul2 =  torch.mean((outputs_l-ul) * (outputs_l-ul), dim=0)
        uu2 =  torch.mean((outputs_u-uu) * (outputs_u-uu), dim=0)
        Su = torch.diag(uu2)
        Sl = torch.diag(ul2).cuda()
        Su_inv = torch.diag(1./(uu2+1e-12)).cuda()
        uul = uu - ul
        detSu = torch.prod(uu2/ul2)
        # loss_dist = (torch.log(torch.det(Su)/(torch.det(Sl)+1e-12)) - Sl.shape[1] + (Su_inv@Sl).trace() + uul.T @ Su_inv @ uul) * 0.5
        loss_dist = (torch.log(detSu) - Sl.shape[1] + (Su_inv@Sl).trace() + uul.T @ Su_inv @ uul) * 0.5
        loss = self.criterion(outputs, labels[labeled_idx])
        alpha = 0.5
        loss = (1-alpha) * loss + alpha * loss_dist
        
        preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
        
        acc += torch.sum(preds == torch.argmax(target, dim=1))
        total += preds.size(0)
        loss.backward()
        
        self.optimizer.step()
        
        return acc, total, loss
    
    def _train(self, epoch):
        running_loss = 0.0
        self.model.train()
        
        acc, total = 0, 0

        print('epoch', epoch)

        tbar = tqdm(range(epoch))

        mse_loss = nn.MSELoss()
        for i in tbar:
            K = int(math.ceil(self.X.shape[0] / self.split_cnt))
            for k in range(K):
                start = k * self.split_cnt
                stop = min((start + self.split_cnt), self.X.shape[0])
                X = self.X[start:stop]
                labels = self.labels[start:stop]
                targets = self.targets[start:stop]
                labeled_idx = self.labeled_idx[(self.labeled_idx >= start) & (self.labeled_idx < stop)]
                labeled_idx = (labeled_idx % self.split_cnt).long()
                unlabeled_idx = self.unlabeled_idx[(self.unlabeled_idx >= start) & (self.unlabeled_idx < stop)]
                unlabeled_idx = (unlabeled_idx % self.split_cnt).long()
                # print('_train_subgraphX', X.shape, labeled_idx.shape, unlabeled_idx.shape, labels.shape, targets.shape)
                acc_, total_, loss = self._train_subgraph(X, labeled_idx, unlabeled_idx, labels, targets)
                acc += acc_
                total += total_
                
            if i % 10 == 0:
                tbar.set_description("Training Label loss {0:.5f}, LR {1:.6f}".format(loss.item(), self.optimizer.param_groups[0]["lr"]))

        print("[Epoch: {}, numImages: {}, numClasses: {}]".format(epoch, total, self.classes))
        print("Training Label Accuracy: {0:.4f}".format(float(acc)/total))


    def _pred(self):
        running_loss = 0.0
        self.model.eval()
        outputs_list = []
        acc, total = 0, 0
        K = int(math.ceil(self.X.shape[0] / self.split_cnt))
        for k in range(K):
            start = k * self.split_cnt
            stop = min((start + self.split_cnt), self.X.shape[0])
            X = self.X[start:stop]
            targets = self.targets[start:stop]
            outputs = self.model(X).detach()
            preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
            acc += torch.sum(preds == torch.argmax(targets, dim=1))
            total += preds.size(0)
            outputs_list.append(outputs)

        print("Label Accuracy: {0:.4f}".format(float(acc)/total))

        outputs = torch.cat(outputs_list, dim=0)
        
        return outputs.detach().cpu().numpy()