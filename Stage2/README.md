# Stage 2

# Dependencies (conda)
```sh
$ conda env create -f environment.yml
$ conda activate relab
```

# How to run
Put the result of Stage1 into root weightsiMix.
```
python -u main_subset.py --net wideresnet282 --epochs 60 --dataset imblancecifar10 --save-dir imbcifar10_imbfactor_10 --batch-size 512 --load weightsiMix/IMBCIFAR10_10_wideresnet282/best_model.pth.tar --lr 0.1 --diffuse --spc 500 --seed 1 --boot-spc 50 --imb_factor 0.1 --beta 10
```

# Dataset Path
You can edited mypath.py to set the path of dataset.
