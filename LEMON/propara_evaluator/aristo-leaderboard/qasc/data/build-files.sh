#!/bin/bash

set -euo pipefail

if [[ ! -f qasc_dataset.tar.gz  ]]; then
  echo Missing file qasc_dataset.tar.gz
  echo
  echo Download it first: http://data.allenai.org/downloads/qasc/qasc_dataset.tar.gz
  exit 1
fi

# Questions with correct answers for train and dev (test set is hidden)
tar -zxvOf qasc_dataset.tar.gz QASC_Dataset/train.jsonl > train.jsonl
tar -zxvOf qasc_dataset.tar.gz QASC_Dataset/dev.jsonl > dev.jsonl

# Predicted answers for train, dev and test (always "A").
tar -zxvOf qasc_dataset.tar.gz QASC_Dataset/train.jsonl | jq -r '[.id, "A"] | @csv' > train-predictions.csv
tar -zxvOf qasc_dataset.tar.gz QASC_Dataset/dev.jsonl | jq -r '[.id, "A"] | @csv' > dev-predictions.csv
tar -zxvOf qasc_dataset.tar.gz QASC_Dataset/test.jsonl | jq -r '[.id, "A"] | @csv' > test-predictions.csv
