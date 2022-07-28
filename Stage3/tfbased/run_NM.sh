export ML_DATA=data
export PYTHONPATH=.
# Download datasets
# CUDA_VISIBLE_DEVICES=0 ./scripts/create_datasets.py
# # Create unlabeled datasets
# CUDA_VISIBLE_DEVICES=0 scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord 

# # Create semi-supervised subsets
# CUDA_VISIBLE_DEVICES=0 scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../imbcifar10_imbfactor_100/seed1/labels_seed1_beta_10.0.npz --labeled ../imbcifar10_imbfactor_100/seed1/subset_seed1_beta_10.0.npz  --name cifar10.imbfac100_10_seed1


# # The experiments will run for 256 epochs
# #250spc
# CUDA_VISIBLE_DEVICES=0 python -u cta/cta_newmatch.py --K=8 --dataset=cifar100.imbfac50_30_seed1 --imb_fac=50 --ema_beta=0.4 --train_dir ./experiments/newmatch/cifar100-imb50_beta30/



CUDA_VISIBLE_DEVICES=0 python -u cta/cta_newmatch.py --K=8 --dataset=cifar10.imbfac100_10_seed1 --imb_fac=100 --ema_beta=0.4 --train_dir ./experiments/newmatch/newlam/
