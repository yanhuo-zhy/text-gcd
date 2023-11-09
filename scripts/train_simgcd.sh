#!/bin/bash
#SBATCH -p long-disi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -t 2-0
#SBATCH -N 1
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100.80:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp_simgcd_cifar100_vith.txt
module load cuda/12.1
source /home/zhun.zhong/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python SimGCD/train.py 