#!/usr/bin/env bash

wget https://github.com/terryqj0107/GECOR/raw/master/CamRest676_for_coreference_and_ellipsis_resolution/CamRest676_annotated.json
python ../../preprocess.py --dataset Task