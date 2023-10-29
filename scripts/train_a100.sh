#!/bin/bash
#SBATCH -p long-disi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100.40:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp_cub0.txt
module load cuda/12.1
source /home/zhun.zhong/miniconda3/bin/activate zhy
CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='cub' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.35\
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --experiment_name='cub_pseudoratio(0.6)_textaug_lambda(0.35)'

CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='cub' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.3\
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --experiment_name='cub_pseudoratio(0.6)_textaug_lambda(0.3)'