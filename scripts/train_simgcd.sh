#!/bin/bash
#SBATCH -p long-disi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -t 2-0
#SBATCH -N 1
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100.40:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp_simgcd_cub_prop_trainlabels2.txt
module load cuda/12.1
source /home/zhun.zhong/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python SimGCD/train.py \
 --dataset_name='cub' \
 --seed_num=2 \
 --prop_train_labels=0.1 \
 --prop_knownclass=0.5 \
 --exp_name='SimGCD_prob_trainlabels(0.1)_seed2' \
 --print_freq=20

CUDA_VISIBLE_DEVICES=0 python SimGCD/train.py \
 --dataset_name='cub' \
 --seed_num=2 \
 --prop_train_labels=0.2 \
 --prop_knownclass=0.5 \
 --exp_name='SimGCD_prob_trainlabels(0.2)_seed2' \
 --print_freq=20

CUDA_VISIBLE_DEVICES=0 python SimGCD/train.py \
 --dataset_name='cub' \
 --seed_num=2 \
 --prop_train_labels=0.3 \
 --prop_knownclass=0.5 \
 --exp_name='SimGCD_prob_trainlabels(0.3)_seed2' \
 --print_freq=20

CUDA_VISIBLE_DEVICES=0 python SimGCD/train.py \
 --dataset_name='cub' \
 --seed_num=2 \
 --prop_train_labels=0.4 \
 --prop_knownclass=0.5 \
 --exp_name='SimGCD_prob_trainlabels(0.4)_seed2' \
 --print_freq=20