#!/bin/bash
#SBATCH --account cvl
#SBATCH -p general
#SBATCH --qos normal
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/text-gcd/temp/temp_cub0.txt

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/pszzz/miniconda3/bin/activate zhy
# source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python kmeans.py \
 --dataset_name='cub' \
 --experiment_name='cub_two_kmeans'

CUDA_VISIBLE_DEVICES=0 python kmeans.py \
 --dataset_name='cifar100' \
 --experiment_name='cifar100_two_kmeans'