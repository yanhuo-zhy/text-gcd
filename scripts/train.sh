#!/bin/bash
###
 # @Author: yanhuo 1760331284@qq.com
 # @Date: 2023-11-11 16:01:59
 # @LastEditors: yanhuo 1760331284@qq.com
 # @LastEditTime: 2024-03-02 16:11:47
 # @FilePath: \text-gcd\scripts\train.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
#SBATCH --account cvl
#SBATCH -p general
#SBATCH --qos normal
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/text-gcd/temp/temp_eccv_scars_wo_conloss.txt

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/pszzz/miniconda3/bin/activate zhy

# CUDA_VISIBLE_DEVICES=0 python get_tag_blip.py
CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='scars' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=1 \
 --interrupted_path='' \
 --batch_size=128 \
 --prop_train_labels=0.5 \
 --experiment_name='eccv_scars_wo_conloss'
