#!/usr/bin/env bash
export model_file=checkpoints_sparc/sparc_concat_none_model
python -m allennlp.service.server_simple \
    --archive-path ${model_file}/model.tar.gz \
    --predictor sparc \
    --include-package predictor.sparc_predictor \
    --include-package dataset_reader.sparc_reader \
    --include-package models.sparc_parser \
    --title "Contextual Semantic Parsing Demo" \
    --field-name question \
    --field-name database_id
