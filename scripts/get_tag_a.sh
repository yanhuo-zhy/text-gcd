#!/bin/bash
#SBATCH -p long-disi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100.40:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp1.txt

module load cuda/12.1
source /home/zhun.zhong/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python get_tag.py --dataset_name='pets'