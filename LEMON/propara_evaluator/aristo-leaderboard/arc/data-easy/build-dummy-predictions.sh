#!/bin/bash

set -euo pipefail

if [[ ! -f ARC-V1-Feb2018.zip ]]; then
  echo Missing file ARC-V1-Feb2018.zip.
  echo
  echo Download it first: https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip
  exit 1
fi

unzip -p ARC-V1-Feb2018.zip ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl | jq -r -c '[.id, .question.choices[0].label] | @csv' | tr -d '"' > dummy-predictions.csv
