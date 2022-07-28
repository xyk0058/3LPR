export ML_DATA=data
export PYTHONPATH=.
# Download datasets
CUDA_VISIBLE_DEVICES=0 ./scripts/create_datasets.py
# Create unlabeled datasets
CUDA_VISIBLE_DEVICES=0 scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord 

# Create semi-supervised subsets
CUDA_VISIBLE_DEVICES=0 scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord --labels ../imbcifar100_imbfactor_10/seed1/labels_seed1_beta_10.0.npz --labeled ../imbcifar100_imbfactor_10/seed1/subset_seed1_beta_10.0.npz  --name cifar100.imbfac10_10_seed1


# The experiments will run for 256 epochs
CUDA_VISIBLE_DEVICES=0 python cta/cta_newmixmatch.py --filters=32 --dataset=cifar100.imbfac20_30_seed1 --w_match=75 --beta=0.75 --imb_fac=20 --ema_beta=0.4  --train_dir ./experiments/newmixmatch/imbcifar100_imb20_beta30/