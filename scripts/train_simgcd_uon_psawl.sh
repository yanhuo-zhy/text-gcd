#!/bin/bash
#SBATCH --account cvl
#SBATCH -p general
#SBATCH --qos normal
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/psawl/hyzheng/text-gcd/temp/temp_simgcd_cub_prop_knowclass2.txt

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python SimGCD/train.py \
 --dataset_name='cub' \
 --seed_num=1 \
 --prop_train_labels=0.5 \
 --prop_knownclass=0.1 \
 --exp_name='True_SimGCD_cub_prob_knowclass(0.1)_seed1' \
 --print_freq=20

CUDA_VISIBLE_DEVICES=0 python SimGCD/train.py \
 --dataset_name='cub' \
 --seed_num=1 \
 --prop_train_labels=0.5 \
 --prop_knownclass=0.2 \
 --exp_name='True_SimGCD_cub_prob_knowclass(0.2)_seed1' \
 --print_freq=20