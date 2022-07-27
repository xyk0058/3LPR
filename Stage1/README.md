# Unofficial Pytorch implementation for *iMix: A Strategy for regularizing Contrastive Representation Learning*.
Paper: https://openreview.net/pdf?id=T6AxtOaWydQ
Code based on https://github.com/PaulAlbert31/iMix

## Requirements
The environment.yml file contains the required packages. You can install them in a new conda environment as follows:
```
conda env create -f environment.yml
conda activate iMix
```

## Run
Run an experiment for cifar100 on GPU0:
```
CUDA_VISIBLE_DEVICES=0 python main.py --save-dir CIFAR100_resnet18 --net resnet18 --dataset cifar100
```
Resume training
```
CUDA_VISIBLE_DEVICES=0 python main.py --save-dir CIFAR100_resnet18 --net resnet18 --dataset cifar100 --resume CIFAR100_resnet18/last_model.pth.tar
```

More can see in run.sh.

## Cite the original paper
```
@inproceedings{2021_ICLR_iMix,
  title="{i-Mix: A Strategy for Regularizing Contrastive Representation Learning}",
  author="Lee, Kibok and Zhu, Yian and Sohn, Kihyuk and Li, Chun-Liang and Shin, Jinwoo and Lee, Honglak",
  booktitle="{International Conference on Learning Representations (ICLR)}",
  year="2021"}
```
