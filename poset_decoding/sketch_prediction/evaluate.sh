#!/usr/bin/env bash
split=mcd1
data_path=./data/$split/
key=$split-sketch
model_path=./model/sketch_prediction-$key
output_file=./output/$key-output
echo $output_file
WORK_DIR=$(readlink -f "./")/sketch_prediction/
echo $WORK_DIR


CUDA_VISIBLE_DEVICES=5 python3 $WORK_DIR/main.py \
--src_path $data_path/train/train_encode.txt --trg_path $data_path/train/train_sketch.txt \
--src_vocabulary $data_path/vocab.cfq.tokens.src  --trg_vocabulary $data_path/vocab.cfq.tokens.sketch \
--embedding_size 300 --batch_size 1 --validate_batch_size 1 \
--save_path $model_path/ --save_interval 500 --log_interval 500 --cuda \
--validation_src_path $data_path/test/test_encode.txt --validation_trg_path $data_path/test/test_sketch.txt  \
--inference_output $model_path/test --type inference \
--model_init_path $model_path/parser_model_best.pt \
--inference_output $output_file

