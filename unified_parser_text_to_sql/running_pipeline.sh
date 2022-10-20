#!/bin/bash

#requirement:
#./data/spider
#./BART-large

# data/spider -> data/spider_schema_linking_tag
python step1_schema_linking.py --dataset=spider

# data/spider_schema_linking_tag -> dataset_post/spider_sl
python step2_serialization.py

###training
python train.py \
  --dataset_path ./dataset_post/spider_sl/bin/ \
  --exp_name spider_sl_v1 \
  --models_path ./models \
  --total_num_update 10000 \
  --max_tokens 1024 \
  --bart_model_path ./data/BART-large \

###evaluate
python step3_evaluate --constrain