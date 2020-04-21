#!/usr/bin/env bash
export model_file=checkpoints_sparc/sparc_concat_none_model
export validation_file=dataset_sparc/dev.json
export validation_out_file=dataset_sparc/dev.jsonl
export prediction_out_file=predict.jsonl
python postprocess.py --valid_file ${validation_file} --valid_out_file ${validation_out_file}
allennlp predict \
--include-package dataset_reader.sparc_reader \
--include-package models.sparc_parser \
--include-package predictor.sparc_predictor \
--predictor sparc \
--dataset-reader-choice validation \
--batch-size 1 \
--cuda-device 0 \
--output-file ${model_file}/${prediction_out_file} \
${model_file}/model.tar.gz ${validation_out_file}