#!/bin/bash
#SBATCH -A IscrC_HDSCisLa
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=pets_textgcd
#SBATCH -o /leonardo_work/IscrC_Fed-GCD/hyzheng/text-gcd/pets_seed0.log

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate textgcd

CUDA_VISIBLE_DEVICES=0 python train_knownclass.py \
 --dataset_name='pets' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=0 \
 --interrupted_path='' \
 --batch_size=128 \
 --prop_train_labels=0.5 \
 --prop_knownclass=0.5 \
 --experiment_name='pets_seed0' \
 --output_dir 'exp_iscap'