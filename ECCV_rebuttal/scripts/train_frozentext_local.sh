#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python frozen_text_baseline.py \
 --dataset_name='cub' \
 --pseudo_ratio=0.6 \
 --lambda_loss=0.2 \
 --coteaching_epoch_t=10 \
 --coteaching_epoch_i=15 \
 --seed_num=1 \
 --interrupted_path='' \
 --batch_size=128 \
 --prop_train_labels=0.5 \
 --experiment_name='cub_frozentext_text(clusterloss)' \
 --output_dir='./rebuttal/train_classnums' \