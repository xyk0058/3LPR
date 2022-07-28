import json
import numpy as np

with open("./data/SSL2/cifar10-unlabel.json",'r') as load_f:
    load_dict = json.load(load_f)
    indexes_u = load_dict['indexes']
    print(len(indexes_u))
with open("./data/SSL2/cifar10.imbfac100_10_seed1_new-label.json",'r') as load_f:
    load_dict = json.load(load_f)
    indexes_l = load_dict['label']
    print(len(indexes_l))

select_idx = np.load('../select_idx_imix.npz')['arr_0']
print('select_idx', select_idx.shape)
ori_label = dict()
for i, ii in enumerate(select_idx):
    ori_label[ii] = i

c = []
for idx in indexes_u:
    if ori_label[idx] in indexes_l:
        c.append(idx)
print(len(c))
