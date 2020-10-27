#!/usr/bin/env bash

split=mcd1
data_path=../data/$split/
key=$split-sketch
model_path=../model/sketch_prediction-$key
output_file=train_log-$key

echo $output_file

mkdir $model_path
CUDA_VISIBLE_DEVICES=4 python3 main.py \
--src_path $data_path/train/train_encode.txt --trg_path $data_path/train/train_sketch.txt \
--src_vocabulary $data_path/vocab.cfq.tokens.src  --trg_vocabulary $data_path/vocab.cfq.tokens.sketch \
--embedding_size 300 --batch_size 64 --validate_batch_size 64 \
--save_path $model_path/ --save_interval 500 --log_interval 500 --cuda \
--iterations 100 \
--validation_src_path $data_path/dev/dev_encode.txt --validation_trg_path $data_path/dev/dev_sketch.txt \
> $output_file

