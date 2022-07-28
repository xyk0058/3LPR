export ML_DATA=data
export PYTHONPATH=.
# Download datasets
CUDA_VISIBLE_DEVICES=0 ./scripts/create_datasets.py
# Create unlabeled datasets
CUDA_VISIBLE_DEVICES=0 scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord 

# Create semi-supervised subsets
CUDA_VISIBLE_DEVICES=0 scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../test_imbcifar10_100_10/seed1/labels_seed1_beta_10.0.npz --labeled ../test_imbcifar10_100_10/seed1/subset_seed1_beta_10.0.npz  --name cifar10.imbfac100_10_seed1_new

# The experiments will run for 256 epochs
#250spc
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.imbfac100_10_seed1_new --train_dir ./experiments/remixmatch/


# Extract accuracy
# python scripts/extract_accuracy.py experiments/remixmatch-c10-4-seed1/cifar10.d.d.d.4spc.seed1@50c-1/CTAugment_depth2_th0.80_decay0.990/CTAReMixMatch_K8_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/

# python scripts/extract_accuracy.py experiments/cifar10_imbfac10_10_seed1_new/cifar10.d.d.d_imbfac100_10_seed1_new-1/CTAugment_depth2_th0.80_decay0.990/CTAReMixMatch_K8_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/