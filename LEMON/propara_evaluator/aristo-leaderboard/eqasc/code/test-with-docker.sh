#!/bin/bash

set -e

echo ------------------------
echo Building evaluator image
echo ------------------------
echo

set -x
docker build -t eqasc-evaluator .
set +x

echo
echo ------------------------
echo Running evaluator on known predictions and labels
echo ------------------------
echo

tempdir=$(mktemp -d /tmp/temp.XXXX)

set -x
docker run \
  -v $PWD/predictions:/predictions:ro \
  -v $PWD/../data:/labels:ro \
  -v $tempdir:/output:rw \
  --entrypoint python \
  eqasc-evaluator \
  allennlp_reasoning_explainqa/evaluator/evaluator.py \
  /predictions/grc.test.predict \
  /labels/chainid_to_label_test.json \
  /output/metrics.json
set +x
echo

echo
echo ------------------------
echo Comparing metrics.json to expected scores
echo ------------------------
echo

echo -n '{"auc_roc": 0.8457533894216488, "explainP1": 0.5387978142076503, "explainNDCG": 0.6376201537170901}' > $tempdir/metrics.json-expected

echo "Expected metrics:"
echo
cat $tempdir/metrics.json-expected
echo
echo

echo "Actual metrics:"
echo
cat $tempdir/metrics.json
echo
echo

echo Diff:
echo
diff -u $tempdir/metrics.json $tempdir/metrics.json-expected

echo "👍 No difference detected. The calculated metrics match the expected ones!"
echo

