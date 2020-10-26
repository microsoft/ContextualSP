#!/usr/bin/env bash
wget https://ai.tencent.com/ailab/nlp/en/dialogue/datasets/Restoration-200K.zip
unzip -j Restoration-200K.zip
rm -rf Restoration-200K.zip
python ../../preprocess.py --dataset Multi