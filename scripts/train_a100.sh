#!/bin/bash
#SBATCH -p long-disi
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --mem=30000
#SBATCH --gres=gpu:a100.80:1
#SBATCH -o /home/zhun.zhong/hyzheng/text-gcd/temp/temp0.txt
