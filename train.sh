#!/bin/bash

formatted_date=$(date +'%b-%d_%H-%M')
echo "The formatted date is: $formatted_date"

python3 train.py \
    --output_dir output/training/$formatted_date/ \
    --model_name_or_path sentence-transformers/all-mpnet-base-v2 \
    --train_file data/keywords.csv \
    --logging_dir logs/training/$formatted_date/ \
    --logging_steps 10 \
    --overwrite_output_dir \
    --per_device_train_batch_size 128 \
    --seed 1016 \
    --num_train_epochs 2 \
    --save_strategy steps \
    --save_steps 1000 \
    --use_labels False \
    --bf16 \
    --learning_rate 5e-5 \
    