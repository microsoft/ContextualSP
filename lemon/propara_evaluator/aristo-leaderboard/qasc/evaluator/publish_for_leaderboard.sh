#!/bin/bash

# This script will test the evaluator, build a docker image, and publish it as
# a Beaker image owned by the Leaderboard user. This is meant to be run by AI2
# after making changes to the QASC evaluator.

set -e

echo --------------------
echo Unit tests
echo --------------------
echo

set -x
python3 test_evaluator.py
set +x

echo
echo --------------------
echo Test docker image
echo --------------------
echo

set -x
./test.sh
set +x

echo
echo --------------------
echo Build local docker image
echo --------------------
echo

NAME="qasc-evaluator-$(date +"%Y%m%d-%H%M%S")"

set -x
docker build -t $NAME .
set +x

echo
echo --------------------
echo Publish Beaker image
echo --------------------
echo

# Beaker must be configured to run as the leaderboard user.
cat >>/tmp/beaker-leaderboard-config.yml <<EOF
agent_address: https://beaker.org
user_token: $(vault read -field=token secret/ai2/alexandria/beaker/dev)
EOF

set -x
export BEAKER_CONFIG_FILE=/tmp/beaker-leaderboard-config.yml

if [[ "$(beaker configure test | grep 'Authenticated as user:')" == 'Authenticated as user: "leaderboard" (us_s03ci03mnt6u)' ]]; then
  echo 'beaker is correctly configured for user "leaderboard"'
else
  echo 'beaker must be configured for user "leaderboard"'
  exit 1
fi

beaker image create -n $NAME $NAME
set +x

rm /tmp/beaker-leaderboard-config.yml
