#!/bin/bash
#SBATCH --account cvl
#SBATCH -p amp20 
#SBATCH --qos amp20  
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/psawl/hyzheng/text-gcd/temp/temp_cifar100_konwnclass2.txt

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train_knownclass.py \
 --dataset_name='cifar100' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=2 \
 --interrupted_path='' \
 --batch_size=128 \
 --prop_train_labels=0.5 \
 --prop_knownclass=0.5 \
 --experiment_name='cifar100_knownclass_0.5_seed2'