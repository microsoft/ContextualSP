#!/bin/bash

set -euo pipefail

if [[ ! -f OpenBookQA-V1-Sep2018.zip  ]]; then
  echo Missing file OpenBookQA-V1-Sep2018.zip
  echo
  echo Download it first: https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip
  exit 1
fi

unzip -p OpenBookQA-V1-Sep2018.zip OpenBookQA-V1-Sep2018/Data/Main/test.jsonl | jq -r -c '{"id":.id, "answerKey":.answerKey}' > question-answers.jsonl
