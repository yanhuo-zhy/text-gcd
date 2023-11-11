#!/bin/bash 
#SBATCH -p long-disi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -t 2-0
#SBATCH -N 1
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100.80:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp_cifar100_vith_fix_1.txt
module load cuda/12.1
source /home/zhun.zhong/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train_vith.py \
 --dataset_name='cifar100' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=20 \
 --coteaching_epoch_i=20 \
 --seed_num=1 \
 --interrupted_path='' \
 --batch_size=110 \
 --experiment_name='cifar100_vith_fixbacbone_20-20'