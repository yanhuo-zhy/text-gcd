#!/bin/bash
#SBATCH --account cvl
#SBATCH -p general 
#SBATCH --qos normal  
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/psawl/hyzheng/text-gcd/GCD/temp_gcd_clip_pets.txt

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python GCD/contrastive_learning.py \
 --dataset_name='pets' \
 --exp_name='gcd_pets_clip'