# Stage 3

# Dependencies
```sh
$ conda env create -f env.yml
$ conda activate relab
```
```
randAugment (python3.7 -m pip install git+https://github.com/ildoonet/pytorch-randaugment), (if an error occurs, type apt-get install git)
```

# How to run
```
python ABCremix_mixup.py --manualSeed 1024 --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --out result_nrmm_100_30_seed1024
```
