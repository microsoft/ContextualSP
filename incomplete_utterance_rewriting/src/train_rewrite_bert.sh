#!/usr/bin/env bash
export model_file=../checkpoints/run_rewrite_bert_
export config_file=../configs/rewrite_bert.jsonnet
export train_data_path=../dataset/Rewrite/train.txt
export validation_data_path=../dataset/Rewrite/dev.txt
export seed=1
allennlp train -s ${model_file} ${config_file} \
--include-package data_reader \
--include-package model \
-o "{\"random_seed\":\"${seed}\",\"numpy_seed\":\"${seed}\",\"pytorch_seed\":\"${seed}\", \"train_data_path\":\"${train_data_path}\",\"validation_data_path\":\"${validation_data_path}\"}"