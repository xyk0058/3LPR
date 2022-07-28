# export ML_DATA=data
# export PYTHONPATH=.
# # Download datasets
# CUDA_VISIBLE_DEVICES=0 ./scripts/create_datasets.py
# # Create unlabeled datasets
# CUDA_VISIBLE_DEVICES=0 scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord 

# # # Create semi-supervised subsets
# # CUDA_VISIBLE_DEVICES=0 scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord --labels ../imbcifar100_imbfactor_10/seed1/labels_seed1_beta_10.0.npz --labeled ../imbcifar100_imbfactor_10/seed1/subset_seed1_beta_10.0.npz  --name cifar100.imbfac10_10_seed1


# # The experiments will run for 256 epochs
# #250spc
# # CUDA_VISIBLE_DEVICES=0 python cta/cta_newmixmatch.py --filters=32 --dataset=cifar100.imbfac20_30_seed1 --w_match=75 --beta=0.75 --imb_fac=20 --ema_beta=0.4  --train_dir ./experiments/newmixmatch/imbcifar100_imb20_beta30/
# # CUDA_VISIBLE_DEVICES=0 python cta/cta_newmixmatch.py --filters=32 --dataset=cifar100.imbfac10_10_seed1 --wd=0.04 --w_match=150 --beta=0.75 --imb_fac=10 --ema_beta=0.4  --train_dir ./experiments/newmixmatch/imbcifar100_imb10_beta10/




# # Extract accuracy
# # python scripts/extract_accuracy.py experiments/newmatch/newlam/cifar10.d.d.d.imbfac100_10_seed1_new-1/CTAugment_depth2_th0.80_decay0.990/CTANewMatch_K8_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/

# # python scripts/extract_accuracy.py experiments/cifar10_imbfac10_10_seed1_new/cifar10.d.d.d_imbfac100_10_seed1_new-1/CTAugment_depth2_th0.80_decay0.990/CTAReMixMatch_K8_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/


# STL10
export ML_DATA=data
export PYTHONPATH=.
# Download datasets
CUDA_VISIBLE_DEVICES=0 ./scripts/create_datasets.py
# Create unlabeled datasets
CUDA_VISIBLE_DEVICES=0 scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord

# # Create semi-supervised subsets
CUDA_VISIBLE_DEVICES=0 scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord --labels ../imbstl10_imbfactor_10/seed1/labels_seed1_beta_90.0.npz --labeled ../imbstl10_imbfactor_10/seed1/subset_seed1_beta_90.0.npz  --name stl10.imbfac10_90_seed1


# The experiments will run for 256 epochs
#250spc
CUDA_VISIBLE_DEVICES=0 python cta/cta_newmixmatch.py --batch=64 --filters=32 --dataset=stl10.imbfac10_90_seed1 --beta=0.75 --w_match=1.5 --imb_fac=10 --ema_beta=0.4 --train_dir ./experiments/newmixmatch/imbstl10_imb10_beta90/
# CUDA_VISIBLE_DEVICES=0 python cta/cta_newmixmatch.py --filters=32 --dataset=cifar100.imbfac10_10_seed1 --wd=0.04 --w_match=150 --beta=0.75 --imb_fac=10 --ema_beta=0.4  --train_dir ./experiments/newmixmatch/imbcifar100_imb10_beta10/




# Extract accuracy
# python scripts/extract_accuracy.py experiments/newmatch/newlam/cifar10.d.d.d.imbfac100_10_seed1_new-1/CTAugment_depth2_th0.80_decay0.990/CTANewMatch_K8_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/

# python scripts/extract_accuracy.py experiments/cifar10_imbfac10_10_seed1_new/cifar10.d.d.d_imbfac100_10_seed1_new-1/CTAugment_depth2_th0.80_decay0.990/CTAReMixMatch_K8_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/
