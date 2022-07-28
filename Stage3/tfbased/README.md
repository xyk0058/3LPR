# This is a slightly modified version of the official implementation of ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring. The original code can be found at https://github.com/google-research/remixmatch. Refer to the README.md at the root of this project for usage.

# Dependencies (conda)
```sh
$ conda env create -f environment.yml
$ conda activate tf
```

# How to run
```
export ML_DATA=data
export PYTHONPATH=.
# Download datasets
CUDA_VISIBLE_DEVICES=0 ./scripts/create_datasets.py
# Create unlabeled datasets
CUDA_VISIBLE_DEVICES=0 scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord 

# Create semi-supervised subsets
# --labels and --labeled are the result of Stage2
CUDA_VISIBLE_DEVICES=0 scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord --labels ../imbcifar100_imbfactor_10/seed1/labels_seed1_beta_10.0.npz --labeled ../imbcifar100_imbfactor_10/seed1/subset_seed1_beta_10.0.npz  --name cifar100.imbfac10_10_seed1

# Stage3
CUDA_VISIBLE_DEVICES=0 python cta/cta_newmixmatch.py --filters=32 --dataset=cifar100.imbfac20_30_seed1 --w_match=75 --beta=0.75 --imb_fac=20 --ema_beta=0.4  --train_dir ./experiments/newmixmatch/imbcifar100_imb20_beta30/
```