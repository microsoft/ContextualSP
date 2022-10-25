from tqdm import tqdm, trange
from transformers import AutoModelForSequenceClassification , AutoTokenizer
import torch.nn as nn
import argparse
import copy
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os, sys, time
import json
import string
import re
from collections import Counter
from datasets import Dataset, load_dataset, load_metric
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
from torch.nn.functional import softmax
import datasets
datasets.set_caching_enabled(False)
from parameters16g_es_corpusb import *

def generate_ver_train_iter(pred):
    logits = pred.predictions
    pred_prob = np.exp(logits) / np.sum(np.exp(logits), axis=-1)[:,None].repeat(3, axis=-1)
    # 0 as entailment, 1 & 2 as non-ent
    ent_prob = pred_prob[:,0, None]
    non_ent_prob = pred_prob[:,1, None] + pred_prob[:,2, None]
    prob = np.concatenate([ent_prob, non_ent_prob], axis=-1)
    pred_labels = np.argmax(prob, axis=-1)

    if trainer.args.local_rank in [-1,0]:
        with open(ver_train_iter_path, "w") as f:
            inputs = infer_dataset["input"]
            outputs = infer_dataset["conclusion"]
            preds = infer_dataset["gen_conclusion"]
            for in_, out_, pred, label in zip(inputs, outputs, preds, pred_labels):
                example1 = {"input": in_, "conclusions": [pred,out_], "is_gold": [1 if 1 - int(label) == 1 else 0,1]} # pred
                # example2 = {"input": in_, "conclusion": out_, "is_gold": 1} # gold
                json.dump(example1, f); f.write("\n")
                # json.dump(example2, f); f.write("\n")
            print(len(inputs))
            print(f"\n\n\nSuccess: ver_train_iter.jsonl has been created at {ver_train_iter_path}.\n\n")
    



datasets =  load_dataset('json', data_files={"infer": unlabeled_ver_train_iter_path}, download_mode='force_redownload')


### model setting setting 1
tokenizer = AutoTokenizer.from_pretrained("ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli")  
infer_dataset = datasets["infer"]
encoder_max_length = 512
decoder_max_length = 128

def process_data_to_model_inputs(batch):
    inputs1 = batch["conclusion"]
    inputs2 = batch["gen_conclusion"]
    inputs = tokenizer(inputs1, inputs2, truncation=True, padding="max_length", max_length=encoder_max_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    return batch

            
## map train data

workers = 8
infer_dataset = infer_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    num_proc=workers,
    batch_size=16,
    load_from_cache_file=False, # necessary
)

infer_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])



training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=nli_per_device_eval_batch_size,
    save_strategy="no",
    output_dir=nli_output_dir
)


# model = BartForConditionalGeneration.from_pretrained("./pbart_checkpoints")
model = AutoModelForSequenceClassification.from_pretrained("ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli")

# pretrained bm 3, lp 3, nr 2

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args
)

if __name__ == "__main__":
    # trainer.train()
    # trainer.predict(valid_dataset, ignore_keys=["encoder_last_hidden_state"])
    pred = trainer.predict(infer_dataset, ignore_keys=["encoder_last_hidden_state"])
    generate_ver_train_iter(pred)
    