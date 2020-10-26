#!/usr/bin/env bash
wget https://obj.umiacs.umd.edu/elgohary/CANARD_Release.zip
unzip -j CANARD_Release.zip
rm -rf CANARD_Release.zip
python ../../preprocess.py --dataset CANARD