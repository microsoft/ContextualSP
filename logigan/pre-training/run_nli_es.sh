#!/bin/bash
GPU_NUM=16
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} nli_es.py
#python nli_es.py
