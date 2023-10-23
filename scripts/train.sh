#!/bin/bash
#SBATCH -p amp20
#SBATCH --qos amp20
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/text-gcd/temp/temp2.txt

module load gcc/gcc-10.2.0
module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0

source /home/pszzz/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='cifar100' \
 --pseudo_num=10 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --experiment_name='cifar100_topk10_pseudo(10-15)_cluster0.1_textaug'