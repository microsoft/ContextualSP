#!/usr/bin/env bash

wget https://raw.githubusercontent.com/chin-gyou/dialogue-utterance-rewriter/master/corpus.txt
python ../../preprocess.py --dataset Rewrite