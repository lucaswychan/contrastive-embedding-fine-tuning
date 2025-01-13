#!/bin/bash

use_labels=False
use_role_graph_data=True

python3 train.py \
    --output_dir output/training/ \
    --model_name_or_path sentence-transformers/all-mpnet-base-v2 \
    --train_file data/keywords.csv \
    --logging_dir logs/training/ \
    --logging_steps 10 \
    --overwrite_output_dir \
    --per_device_train_batch_size 128 \
    --seed 1016 \
    --num_train_epochs 3 \
    --save_strategy steps \
    --save_steps 1500 \
    --use_labels $use_labels \
    --use_role_graph_data $use_role_graph_data \
    --bf16 \
    --learning_rate 5e-5 \
    