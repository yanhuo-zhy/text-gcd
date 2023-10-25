#!/bin/bash
#SBATCH -p cs
#SBATCH --qos csstaff 
#SBATCH --account cs 
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=15000
#SBATCH --gres gpu:1
#SBATCH -o /home/psawl/hyzheng/text-gcd/temp/temp_cifar0.txt

module load gcc/gcc-10.2.0
module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0

# source /home/pszzz/miniconda3/bin/activate zhy
source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='cifar100' \
 --pseudo_ratio=0.4 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --experiment_name='cifar100_pseudoratio(0.4)_textaug'