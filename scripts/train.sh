#!/bin/bash
#SBATCH --account cvl
#SBATCH -p amp48
#SBATCH --qos amp48
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=15000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/text-gcd/temp/temp_cub_vith.txt

module load gcc/gcc-10.2.0
module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0

source /home/pszzz/miniconda3/bin/activate zhy
# source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='cub' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=1 \
 --interrupted_path='' \
 --experiment_name='cub_vith16_test'