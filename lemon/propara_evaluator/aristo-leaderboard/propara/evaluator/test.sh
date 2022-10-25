#!/bin/bash

set -e

export PYTHONPATH=.

echo
echo ----------------------------------
echo unit tests
echo ----------------------------------
echo

set -x

pytest

set +x

echo
echo ----------------------------------
echo mypy
echo ----------------------------------
echo

set -x

mypy $(find . -type f -name '*.py')

echo "Hurray, mypy didn't find problems with the code."

set +x

echo
echo ----------------------------------
echo testfiles-1
echo ----------------------------------
echo

set -x

python3 evaluator.py -p testfiles-1/predictions.tsv -a testfiles-1/answers.tsv -o /tmp/metrics.json

if [ "$(cat /tmp/metrics.json)" != '{"precision": 0.743, "recall": 0.43, "f1": 0.545}' ]; then
    echo File /tmp/metrics.json looks wrong.
    exit 1
fi

echo File /tmp/metrics.json looks okay.

set +x

echo
echo ----------------------------------
echo testfiles-2
echo ----------------------------------
echo

set -x

python3 evaluator.py -p testfiles-2/predictions.tsv -a testfiles-2/answers.tsv -o /tmp/metrics.json

if [ "$(cat /tmp/metrics.json)" != '{"precision": 1.0, "recall": 1.0, "f1": 1.0}' ]; then
    echo File /tmp/metrics.json looks wrong.
    exit 1
fi

echo File /tmp/metrics.json looks okay.

set +x

echo
echo ----------------------------------
echo testfiles-3
echo ----------------------------------
echo

set -x

python3 evaluator.py -p testfiles-3/predictions.tsv -a testfiles-3/answers.tsv -o /tmp/metrics.json

if [ "$(cat /tmp/metrics.json)" != '{"precision": 0.833, "recall": 0.583, "f1": 0.686}' ]; then
    echo File /tmp/metrics.json looks wrong.
    exit 1
fi

echo File /tmp/metrics.json looks okay.

set +x

echo
echo ----------------------------------
echo testfiles-4
echo ----------------------------------
echo

set -x
set +e

python3 evaluator.py -p testfiles-4/predictions.tsv -a testfiles-4/answers.tsv -o /tmp/metrics.json

exit_status=$?
if [ $exit_status -eq 2 ]; then
    echo "Got expected exit status: $exit_status"
else
    echo "Got unexpected exit status: $exit_status"
    exit 1
fi

set -e
set +x

echo
echo ----------------------------------
echo testfiles-5
echo ----------------------------------
echo

set -x
set +e

python3 evaluator.py -p testfiles-5/predictions.tsv -a testfiles-5/answers.tsv -o /tmp/metrics.json

exit_status=$?
if [ $exit_status -eq 2 ]; then
    echo "Got expected exit status: $exit_status"
else
    echo "Got unexpected exit status: $exit_status"
    exit 1
fi

set -e
set +x

echo
echo ----------------------------------
echo testfiles-6
echo ----------------------------------
echo

set -x
set +e

python3 evaluator.py -p testfiles-6/predictions.tsv -a testfiles-6/answers.tsv -o /tmp/metrics.json

exit_status=$?
if [ $exit_status -eq 2 ]; then
    echo "Got expected exit status: $exit_status"
else
    echo "Got unexpected exit status: $exit_status"
    exit 1
fi

set -e
set +x


echo
echo ----------------------------------
echo testfiles-7
echo ----------------------------------
echo

set -x

python3 evaluator.py -p testfiles-7/predictions.tsv -a testfiles-7/answers.tsv -o /tmp/metrics.json

if [ "$(cat /tmp/metrics.json)" != '{"precision": 0.617, "recall": 0.448, "f1": 0.519}' ]; then
    echo File /tmp/metrics.json looks wrong.
    exit 1
fi

echo File /tmp/metrics.json looks okay.

set +x


echo
echo ----------------------------------
echo Docker
echo ----------------------------------
echo

set -x

./test-in-docker.sh
