#!/bin/bash
GPU_NUM=16
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} hf_generation_multi_es.py \
    --model_name_or_path $1 \
    --output_dir $2 \
    --data_dir $3 \
    --train_file $4 \
    --validation_file $5 \
    --per_device_train_batch_size $6 \
    --gradient_accumulation_steps $7 \
    --learning_rate $8 \
    --num_train_epochs $9 \
    --seed ${10} \
    --remove_unused_columns False \
    --num_beams ${17} \
    --save_strategy epoch \
    --evaluation_strategy no \
    --logging_steps 200 \
    --max_train_samples ${11} \
    --max_predict_samples ${12} \
    --predict_with_generate \
    --do_predict ${13} \
    --test_file ${14} \
    --do_eval False \
    --do_train ${15} \
    --max_eval_samples 16 \
    --prediction_mode ${16} \
    --gan_alpha ${18} \
    --per_device_eval_batch_size ${19} \
    --overwrite_cache\
    --overwrite_output_dir

