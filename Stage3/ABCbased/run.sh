# #!/bin/bash


python ABCremix_mixup.py --manualSeed 0 --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --out result_nrmm_100_30_seed0

python ABCremix.py --manualSeed 0 --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --out result_abcrmm_100_30_seed0

python remixmatch.py --manualSeed 0 --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --out result_rmm_100_30_seed0


python ABCremix_mixup.py --manualSeed 1024 --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --out result_nrmm_100_30_seed1024


python test_ABCremix_mixup.py --manualSeed 0 --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --resume ./result_abcrmm_100_30_seed0/checkpoint.pth.tar




python ABCremix_mixup_ir3.py --manualSeed 0 --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --out result_nrmmir3_100_30_seed0

# python ABCremix_mixup.py --manualSeed 1024 --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --out result_nrmm_100_10_seed1024

# python ABCremix.py --manualSeed 1024 --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --out result_rmm_100_10_seed1024


# python test_ABCremix_mixup.py --manualSeed 1024 --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --resume ./result_nrmm_100_10_seed1024/checkpoint.pth.tar


# python test_ABCremix_mixup.py --manualSeed 1024 --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 100 --epoch 500 --val-iteration 500 --dataset cifar10 --imbalancetype long --resume ./result_rmm_100_10_seed1024/model_300.pth.tar



# python ABCremix_mixup.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 1000 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --out result_nrmm_100_30

# python test_ABCremix_mixup.py --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_nrmm_100_10/checkpoint.pth.tar
# python test_ABCremix_mixup.py --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_rmm_100_10/checkpoint.pth.tar


# python ABCremix_mixup.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 1000 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long



# python ABCfix_mixup.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 1000 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --out result_fm_100_30_labeledmix

# python ABCfix.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --out result_fm_100_30_baseline



# python plot3d.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_remixmatch/checkpoint.pth.tar

# python eval_knn.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_remixmatch/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0 python eval_linear.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_remixmatch/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0 python eval_knn.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_remixmatch/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0 python eval_linear.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./imb100_beta30/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0 python eval_linear.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_fm_100_30_baseline/checkpoint.pth.tar


# CUDA_VISIBLE_DEVICES=0 python eval_knn.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./imb100_beta30/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0 python eval_knn.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 50 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_rmm_50_30/checkpoint.pth.tar


# python ce.py --gpu 0 --label_ratio 100 --num_max 5000 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --out result_ce



# python plot.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_ce/checkpoint.pth.tar


# python remixmatch.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 50 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --out result_rmm_50_30

# python ce.py --gpu 0 --label_ratio 100 --num_max 5000 --imb_ratio 150 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --out result_ce_150_100

# python ce.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --out result_ce_100_30


# CUDA_VISIBLE_DEVICES=0 python eval_knn.py --gpu 0 --label_ratio 100 --num_max 5000 --imb_ratio 50 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --resume ./result_ce/checkpoint.pth.tar
