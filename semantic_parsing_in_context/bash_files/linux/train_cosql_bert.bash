#!/usr/bin/env bash
export seed=1
export config_file=train_configs_bert/concat.none.jsonnet
export model_file=checkpoints_cosql/cosql_bert_concat_none_model
export tables_file=dataset_cosql/tables.json
export database_path=dataset_cosql/database
export dataset_path=dataset_cosql
export train_data_path=dataset_cosql/train.json
export validation_data_path=dataset_cosql/dev.json
allennlp train -s ${model_file} ${config_file} \
--include-package dataset_reader.sparc_reader \
--include-package models.sparc_parser \
-o "{\"model.serialization_dir\":\"${model_file}\",\"random_seed\":\"${seed}\",\"numpy_seed\":\"${seed}\",\"pytorch_seed\":\"${seed}\",\"dataset_reader.tables_file\":\"${tables_file}\",\"dataset_reader.database_path\":\"${database_path}\",\"train_data_path\":\"${train_data_path}\",\"validation_data_path\":\"${validation_data_path}\",\"model.dataset_path\":\"${dataset_path}\"}"