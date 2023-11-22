#!/bin/bash 
#SBATCH -p long-disi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -t 2-0
#SBATCH -N 1
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100.80:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp_cifar100_konwnclass3.txt
module load cuda/12.1
source /home/zhun.zhong/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train_knownclass.py \
 --dataset_name='cifar100' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=0 \
 --interrupted_path='' \
 --batch_size=128 \
 --prop_train_labels=0.5 \
 --prop_knownclass=0.3 \
 --experiment_name='cifar100_knownclass_0.3_seed0'

CUDA_VISIBLE_DEVICES=0 python train_knownclass.py \
 --dataset_name='cifar100' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=0 \
 --interrupted_path='' \
 --batch_size=128 \
 --prop_train_labels=0.5 \
 --prop_knownclass=0.4 \
 --experiment_name='cifar100_knownclass_0.4_seed0'