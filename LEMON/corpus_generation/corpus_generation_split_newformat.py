import json
import sys
import copy
from itertools import combinations, permutations
import math
import argparse
from random import shuffle
from remove_same import big_file_remove_same
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_prefix", type=str, default='alchemy', help="dataset prefix")
parser.add_argument("--root_path", type=str, default='../corpus/', help="dataset prefix")

args = parser.parse_args()

args.corpus_file = os.path.join(args.root_path, '{}/pretraining_corpus_{}.txt'.format(args.dataset_prefix, args.dataset_prefix))
args.remove_same_file = os.path.join(args.root_path, '{}/temp.txt'.format(args.dataset_prefix))
args.train_source_file = os.path.join(args.root_path, '{}/train.src'.format(args.dataset_prefix))
args.train_target_file = os.path.join(args.root_path, '{}/train.tgt'.format(args.dataset_prefix))
args.dev_source_file = os.path.join(args.root_path, '{}/dev.src'.format(args.dataset_prefix))
args.dev_target_file = os.path.join(args.root_path, '{}/dev.tgt'.format(args.dataset_prefix))

big_file_remove_same(args.corpus_file, args.remove_same_file)

with open(args.remove_same_file, 'r') as f:
    total_data_list = f.readlines()

print(len(total_data_list))
shuffle(total_data_list)

train_data_list = total_data_list[:-20000]
dev_data_list = total_data_list[-20000:]

fw_train_src = open(args.train_source_file, 'w')
fw_train_tgt = open(args.train_target_file, 'w')
fw_dev_src = open(args.dev_source_file, 'w')
fw_dev_tgt = open(args.dev_target_file, 'w')

for item in train_data_list:
    try:
        action, prev_state, current_state = item.split('\t')
    except:
        continue
    src_row = ' SEP '.join([prev_state.strip(), action.strip()])
    tgt_row = current_state.strip()
    fw_train_src.write(src_row)
    fw_train_src.write('\n')
    fw_train_tgt.write(tgt_row)
    fw_train_tgt.write('\n')

for item in dev_data_list:
    try:
        action, prev_state, current_state = item.split('\t')
    except:
        continue
    src_row = ' SEP '.join([prev_state.strip(), action.strip()])
    tgt_row = current_state.strip()
    fw_dev_src.write(src_row)
    fw_dev_src.write('\n')
    fw_dev_tgt.write(tgt_row)
    fw_dev_tgt.write('\n')

