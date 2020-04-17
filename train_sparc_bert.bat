set CUDA_VISIBLE_DEVICES=5
set seed=4
set config_file=train_configs_bert/turn.none.jsonnet
set model_file=checkpoints_sparc/sparc_bert_turn_none_model_4
set tables_file=dataset_sparc/tables.json
set database_path=dataset_sparc/database
set dataset_path=dataset_sparc
set train_data_path=dataset_sparc/train.json
set validation_data_path=dataset_sparc/dev.json
allennlp train -s %model_file% %config_file% ^
--include-package dataset_reader.sparc_reader ^
--include-package models.sparc_parser ^
-o {"""model.serialization_dir""":"""%model_file%""","""random_seed""":"""%seed%""","""numpy_seed""":"""%seed%""","""pytorch_seed""":"""%seed%""","""dataset_reader.tables_file""":"""%tables_file%""","""dataset_reader.database_path""":"""%database_path%""","""train_data_path""":"""%train_data_path%""","""validation_data_path""":"""%validation_data_path%""","""model.dataset_path""":"""%dataset_path%"""}