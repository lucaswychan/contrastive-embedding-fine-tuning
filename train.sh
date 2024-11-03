#!/bin/bash

formatted_date=$(date +'%b-%d_%H-%M')
echo "The formatted date is: $formatted_date"

python3 train.py \
    --output_dir output/training/$formatted_time/ \
    --model_name_or_path sentence-transformers/all-mpnet-base-v2 \
    --train_file data/keywords.csv \
    --logging_dir logs/training/$formatted_time/ \
    --logging_steps=10 \
    --overwrite_output_dir \
    --seed 1016 \
    --bf16 \
    --num_train_epochs 3 \
    --save_strategy steps \
    --save_steps 500 \
    