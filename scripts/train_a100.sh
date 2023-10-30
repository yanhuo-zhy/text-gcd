#!/bin/bash
#SBATCH -p long-disi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -t 2-0
#SBATCH -N 1
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100.40:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp_food3.txt
module load cuda/12.1
source /home/zhun.zhong/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='food' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2\
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --experiment_name='food_pseudoratio(0.6)_textaug_lambda(0.2)'