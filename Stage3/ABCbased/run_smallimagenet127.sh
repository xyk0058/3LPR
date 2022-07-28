# #!/bin/bash

python ABCremix_mixup.py --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset smallimagenet127_x32 --imbalancetype long
