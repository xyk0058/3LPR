#!/bin/bash

# imbcifar10 beta10
CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_10 --batch-size 512 --load weightsiMix/IMBCIFAR10_10_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.1 --beta 10 >log/imbcifar10_fac10_beta10.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_20 --batch-size 512 --load weightsiMix/IMBCIFAR10_20_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.05 --beta 10 >log/imbcifar10_fac20_beta10.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_50 --batch-size 512 --load weightsiMix/IMBCIFAR10_50_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.02 --beta 10 >log/imbcifar10_fac50_beta10.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_100 --batch-size 512 --load weightsiMix/IMBCIFAR10_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.01 --beta 10 >log/imbcifar10_fac100_beta10.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_150 --batch-size 512 --load weightsiMix/IMBCIFAR10_150_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.00666667 --beta 10 >log/imbcifar10_fac150_beta10.out 2>&1 &
wait

# imbcifar10 beta30
CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_10 --batch-size 512 --load weightsiMix/IMBCIFAR10_10_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.1 --beta 30 >log/imbcifar10_fac10_beta30.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_20 --batch-size 512 --load weightsiMix/IMBCIFAR10_20_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.05 --beta 30 >log/imbcifar10_fac20_beta30.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_50 --batch-size 512 --load weightsiMix/IMBCIFAR10_50_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.02 --beta 30 >log/imbcifar10_fac50_beta30.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_100 --batch-size 512 --load weightsiMix/IMBCIFAR10_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.01 --beta 30 >log/imbcifar10_fac100_beta30.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_150 --batch-size 512 --load weightsiMix/IMBCIFAR10_150_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.00666667 --beta 30 >log/imbcifar10_fac150_beta30.out 2>&1 &
wait

# imbcifar100 beta10
CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar100 --save-dir imbcifar100_imbfactor_10 --batch-size 512 --load weightsiMix/IMBCIFAR100_10_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.1 --beta 10 >log/imbcifar100_fac10_beta10.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar100 --save-dir imbcifar100_imbfactor_20 --batch-size 512 --load weightsiMix/IMBCIFAR100_20_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.05 --beta 10 >log/imbcifar100_fac20_beta10.out 2>&1 &
wait


CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar100 --save-dir imbcifar100_imbfactor_50 --batch-size 512 --load weightsiMix/IMBCIFAR100_50_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.02 --beta 10 >log/imbcifar100_fac50_beta10.out 2>&1 &
wait


CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar100 --save-dir imbcifar100_imbfactor_100 --batch-size 512 --load weightsiMix/IMBCIFAR100_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.01 --beta 10 >log/imbcifar100_fac100_beta10.out 2>&1 &
wait


# imbcifar100 beta30
CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar100 --save-dir imbcifar100_imbfactor_10 --batch-size 512 --load weightsiMix/IMBCIFAR100_10_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.1 --beta 30 >log/imbcifar100_fac10_beta30.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar100 --save-dir imbcifar100_imbfactor_20 --batch-size 512 --load weightsiMix/IMBCIFAR100_20_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.05 --beta 30 >log/imbcifar100_fac20_beta30.out 2>&1 &
wait


CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar100 --save-dir imbcifar100_imbfactor_50 --batch-size 512 --load weightsiMix/IMBCIFAR100_50_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.02 --beta 30 >log/imbcifar100_fac50_beta30.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar100 --save-dir imbcifar100_imbfactor_100 --batch-size 512 --load weightsiMix/IMBCIFAR100_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.01 --beta 30 >log/imbcifar100_fac100_beta30.out 2>&1 &
wait





# imbstl10 beta90
CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancestl10 --save-dir imbstl10_imbfactor_10 --batch-size 512 --load weightsiMix/IMBSTL10_10_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.1 --beta 90 >log/imbstl10_fac10_beta90.out 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancestl10 --save-dir imbstl10_imbfactor_20 --batch-size 512 --load weightsiMix/IMBSTL10_20_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.1 --beta 90 >log/imbstl10_fac20_beta90.out 2>&1 &






CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_100 --batch-size 512 --load weightsiMix/IMBCIFAR10_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 10 --imb_factor 0.01 --beta 10 >log/imbcifar10_fac100_beta10_select10.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_100 --batch-size 512 --load weightsiMix/IMBCIFAR10_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 100 --imb_factor 0.01 --beta 10 >log/imbcifar10_fac100_beta10_select100.out 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_100 --batch-size 512 --load weightsiMix/IMBCIFAR10_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 100 --imb_factor 0.01 --beta 10 >log/imbcifar10_fac100_beta10_selectm1.5.out 2>&1 &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset2.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_100 --batch-size 512 --load weightsiMix/result_rmm_50_30/checkpoint.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.01 --beta 30 >log/imbcifar10_fac100_beta50_test.out 2>&1 &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 60 --dataset smallimagenet127_x32 --save-dir SMALLIMAGENET127_X32 --batch-size 256 --load weightsiMix/SMALLIMAGENET127x32_wideresnet282_400epoch/last_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.0034965 --beta 10 >log/smallimagenet_x32_beta10.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -u main_subset.py --net wideresnet282 --epochs 60 --dataset smallimagenet127_x32 --save-dir SMALLIMAGENET127_X32 --batch-size 256 --load weightsiMix/SMALLIMAGENET127x32_wideresnet282_400epoch/last_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.0034965 --beta 10






CUDA_VISIBLE_DEVICES=0 python -u main_subset.py --net wideresnet282 --epochs 0 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_100_nodiffuse --batch-size 512 --load weightsiMix/IMBCIFAR10_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 0 --imb_factor 0.01 --beta 10




CUDA_VISIBLE_DEVICES=0 nohup python -u main_subset.py --net wideresnet282 --epochs 1 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_100_nodiffuse --batch-size 512 --load weightsiMix/IMBCIFAR10_100_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 0 --imb_factor 0.01 --beta 30 >log/imbcifar10_fac100_beta30_nodiffuse.out 2>&1 &