#!/bin/bash
#SBATCH --account cs
#SBATCH -p amp48
#SBATCH --qos amp48
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=30000
#SBATCH --gres gpu:1
#SBATCH -o /home/psawl/hyzheng/text-gcd/temp/temp_cifar100_1.txt

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

# source /home/pszzz/miniconda3/bin/activate zhy
source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='cifar100' \
 --pseudo_ratio=0.3 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=1 \
 --interrupted_path='' \
 --batch_size=128 \
 --experiment_name='cifar100_ablation_pseudo_ratio(0.3)'

CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='cifar100' \
 --pseudo_ratio=0.4 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=1 \
 --interrupted_path='' \
 --batch_size=128 \
 --experiment_name='cifar100_ablation_pseudo_ratio(0.4)'