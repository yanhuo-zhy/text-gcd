#!/bin/bash
#SBATCH --account cvl
#SBATCH -p amp48
#SBATCH --qos amp48
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/text-gcd/temp/temp0.txt

module load gcc/gcc-10.2.0
module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0

source /home/pszzz/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python get_tag.py --dataset_name='imagenet'
