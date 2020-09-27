set seed=1
set config_file=train_configs_bert/concat.none.jsonnet
set model_file=checkpoints_cosql/cosql_bert_concat_none_model
set tables_file=dataset_cosql/tables.json
set database_path=dataset_cosql/database
set dataset_path=dataset_cosql
set train_data_path=dataset_cosql/train.json
set validation_data_path=dataset_cosql/dev.json
allennlp train -s %model_file% %config_file% ^
--include-package dataset_reader.sparc_reader ^
--include-package models.sparc_parser ^
-o {"""model.serialization_dir""":"""%model_file%""","""random_seed""":"""%seed%""","""numpy_seed""":"""%seed%""","""pytorch_seed""":"""%seed%""","""dataset_reader.tables_file""":"""%tables_file%""","""dataset_reader.database_path""":"""%database_path%""","""train_data_path""":"""%train_data_path%""","""validation_data_path""":"""%validation_data_path%""","""model.dataset_path""":"""%dataset_path%"""}