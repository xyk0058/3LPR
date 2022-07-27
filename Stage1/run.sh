#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python main.py --save-dir IMBSTL10_10_wideresnet282 --net wideresnet282 --dataset imblancestl10 --imb_factor 0.1 --no-eval

# CUDA_VISIBLE_DEVICES=0 python main.py --save-dir IMBSTL10_20_wideresnet282 --net wideresnet282 --dataset imblancestl10 --imb_factor 0.05 --no-eval

# CUDA_VISIBLE_DEVICES=0 python eval_linear.py IMBCIFAR10_100_wideresnet282/best_model.pth.tar


# CUDA_VISIBLE_DEVICES=0 python main.py --save-dir SMALLIMAGENET127x32_wideresnet282 --net wideresnet282 --dataset ImbSmallImageNet127_x32 --no-eval --batch-size 512 --epochs 800 --resume SMALLIMAGENET127x32_wideresnet282/last_model.pth.tar
