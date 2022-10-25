#!/bin/bash

set -euo pipefail

echo
echo --------------------------------
echo Building image
echo --------------------------------
echo

set -x

docker build -t propara-evaluator-local . 

set +x

echo
echo --------------------------------
echo Running
echo --------------------------------
echo

set -x

T=$(mktemp -d /tmp/tmp-XXXXX)

docker run \
 -v $PWD/testfiles-1:/testfiles-1:ro \
 -v $T:/output:rw \
 -it propara-evaluator-local \
 python3 \
 evaluator.py \
 --predictions /testfiles-1/predictions.tsv \
 --answers /testfiles-1/answers.tsv \
 --output /output/metrics.json

if [ "$(cat $T/metrics.json)" != '{"precision": 0.743, "recall": 0.43, "f1": 0.545}' ]; then
    echo File $T/metrics.json looks wrong.
    exit 1
fi

echo $T/metrics.json looks okay.

set +x
