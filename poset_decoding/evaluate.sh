#!/usr/bin/env bash

## generate sketch
bash ./sketch_prediction/evaluate.sh
## preprocess data for traversal path prediction
python preprocess_hierarchical_inference.py
## generate valid traversal path
python ./traversal_path_prediction/MatchZoo-py/evaluate_esim.py
## evaluate, output accuracy score
python evaluate.py
