#!/usr/bin/env bash
export model_file=../checkpoints/run_canard
export config_file=../configs/canard.jsonnet
export train_data_path=../dataset/CANARD/train.txt
export validation_data_path=../dataset/CANARD/dev.txt
export pretrained_file=../glove/glove.6B.100d.txt
export seed=1
allennlp train -s ${model_file} ${config_file} \
--include-package data_reader \
--include-package model \
-o "{\"random_seed\":\"${seed}\",\"numpy_seed\":\"${seed}\",\"pytorch_seed\":\"${seed}\", \"train_data_path\":\"${train_data_path}\",\"validation_data_path\":\"${validation_data_path}\",\"model.word_embedder.tokens.pretrained_file\":\"${pretrained_file}\"}"