#!/bin/bash

set -xe

docker build -t aristo-leaderboard-eval-test .

T=$(mktemp -d /tmp/tmp-XXXXX)

docker run \
  -v $T:/output:rw \
  -v $PWD:/input:ro \
  aristo-leaderboard-eval-test \
  ./evaluator.py \
  --question-answers /input/questions.jsonl \
  --predictions /input/predictions.csv \
  --output /output/metrics.json

if [ "$(cat $T/metrics.json)" != '{"accuracy": 0.85}' ]; then
    echo File $T/metrics.json looks wrong.
    exit 1
fi

echo File $T/metrics.json looks okay.

