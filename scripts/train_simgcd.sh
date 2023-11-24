#!/bin/bash
#SBATCH -p long-disi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -t 2-0
#SBATCH -N 1
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100.80:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp_simgcd_cifar100_prop_knownclass5.txt
module load cuda/12.1
source /home/zhun.zhong/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python SimGCD/train.py \
 --dataset_name='cifar100' \
 --seed_num=2 \
 --prop_train_labels=0.5 \
 --prop_knownclass=0.2 \
 --exp_name='SimGCD_cifar100_prob_knownclass(0.2)_seed2' \
 --print_freq=20