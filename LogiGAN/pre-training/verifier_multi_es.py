from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn as nn
from copy import deepcopy
from collections import Counter
from datasets import Dataset, load_dataset
import numpy as np
from transformers import Trainer, TrainingArguments
from torch.nn.functional import softmax
from parameters16g_es_corpusb import *
import os, time
import json
import datasets
import random
datasets.set_caching_enabled(False)
os.environ["WANDB_DISABLED"] = "true"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help='Path to load verifier model')
parser.add_argument('--output_dir', help='Path to save new checkpoint after training.')
parser.add_argument('--local_rank', help='local rank')
args = parser.parse_args()
VER_MODEL_PATH = args.model_path
# if not VER_MODEL_PATH.startswith('checkpoint-'):
#     files=os.listdir(VER_MODEL_PATH)
#     for f in files:
#         if f.startswith('checkpoint-'): VER_MODEL_PATH = os.path.join(VER_MODEL_PATH, f)
if args.local_rank == '0': print(f"\nVerifier.py: Using Verifier model from {VER_MODEL_PATH}\n\n")

def compute_metrics(predictions):
    pred_logits, labels = predictions
    # print(pred_logits.shape)
    return {"accuracy": 1}

def create_ver_infer(pred_dataset, pred):
    logits, labels = pred[0][:,:2], pred[1]
    pred_probs = softmax(torch.FloatTensor(logits), dim=-1).numpy()[:, 1]
    # print(len(pred_probs))
    start_idx = 0
    if trainer.args.local_rank in [-1,0]:
        with open(gen_train_iter_path, "w") as f:  # Xinyu: Verifier will create train set for generator at current iteration.
            for i,cs,gs,ids in zip(pred_dataset["input"], pred_dataset["conclusions"], pred_dataset["is_gold"],pred_dataset['selected_ids']):
                ins_num = min(len(cs), gen_per_device_examples_num)
                tmp_scores = [float(p) for p in pred_probs[start_idx:start_idx+ins_num]]
                start_idx += ins_num

                example = {"input": i, "conclusions": cs, "is_gold": gs, "ver_prob": tmp_scores}

                json.dump(example, f)
                f.write("\n")
            print(f"\n\n\nSuccess: gen_infer.jsonl has been created at {gen_train_iter_path}.\n\n\n")


###########################################
data_files = {
    'train': ver_train_iter_path,
    'infer':gen_train_src_path
    # 'infer': unlabeled_gen_train_iter_path
}
datasets =  load_dataset('json', data_files=data_files, download_mode="force_redownload")
train_dataset = datasets["train"]
infer_dataset = datasets["infer"]
# Select random subset
np.random.seed(int(time.time()))
train_dataset = train_dataset.select(np.random.choice(np.arange(len(train_dataset["input"])), ver_train_samples_per_iter, replace=False))
infer_examples = infer_dataset.select(np.random.choice(np.arange(len(infer_dataset["input"])), gen_train_samples_per_iter, replace=False))


#load config
config = AutoConfig.from_pretrained(VER_MODEL_PATH)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(VER_MODEL_PATH, use_fast=True)
special_tokens=['[SEP]','[MASK]']
if not all([t in tokenizer.vocab for t in special_tokens]): # add speical token only if they are not found
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    added_tokens = tokenizer.get_added_vocab()
    print(f"\nAdded special tokens {special_tokens} into tokenizer vocab.\n")


encoder_max_length = 512
def process_infer_dataset_sample(examples):
    all_cs = []
    all_selected_ids = []
    all_gs = []
    for id, (i, cs,gs) in enumerate(zip(examples["input"], examples["conclusions"],examples["is_gold"])):
        ins_num = min(len(cs),gen_per_device_examples_num)-1 #if not do_train else len(cs)
        gold_id = gs.index(1)
        all_ids = list(range(len(gs)))
        all_ids.remove(gold_id)
        selected_ids = list(np.random.choice(all_ids,ins_num,replace=False))
        selected_ids.append(gold_id)
        all_selected_ids.append(selected_ids)
        tmp_cs = [cs[j] for j in selected_ids]
        all_cs.append(tmp_cs)
        all_gs.append([gs[j] for j in selected_ids])
        # print(tmp_cs,[gs[j] for j in selected_ids],selected_ids)
    examples['conclusions'] = all_cs
    examples['is_gold']= all_gs
    examples['selected_ids'] = all_selected_ids
    return examples


def process_data_to_model_inputs(examples):
    inputs = []
    ids = []
    all_selected_ids = []
    for id, (i, cs,gs) in enumerate(zip(examples["input"], examples["conclusions"],examples["is_gold"])):
        for c in cs:
            inputs.append(i.replace("[MASK]", c))
        ids.extend([id]*len(cs))
    # inputs = [i.replace("[MASK]", c) for i,c in zip(examples["input"],examples["conclusion"])]
    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length
    )

    model_inputs = {}
    model_inputs['ids']=ids
    model_inputs["input_ids"] = tokenized_inputs.input_ids
    model_inputs["attention_mask"] = tokenized_inputs.attention_mask
    model_inputs["labels"] = [l for ls in examples["is_gold"] for l in ls]#[ls[index] for j,ls in enumerate(examples["is_gold"]) for index in all_selected_ids[j]]
    # since above lists are references, the following line changes the 0 index for all samples
    # print(model_inputs['labels'])
    return model_inputs
training_args = TrainingArguments(
    # evaluation_strategy="steps",
    per_device_train_batch_size=ver_per_device_train_batch_size,
    per_device_eval_batch_size=ver_per_device_eval_batch_size,
    fp16=False,
    half_precision_backend="amp",
    output_dir=args.output_dir,
    logging_steps=100,
    save_strategy="no",
    gradient_accumulation_steps=ver_gradient_accumulation_steps,
    num_train_epochs=1,
    do_eval=False,
    learning_rate=ver_learning_rate,
    eval_accumulation_steps=1
)
## map train data
column_names = train_dataset.column_names
train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    # do_train=True,
    batched=True,
    remove_columns=column_names,
    num_proc=8,
    batch_size=16,
    load_from_cache_file=False, # necessary
)

infer_examples = infer_examples.map(
    process_infer_dataset_sample,
    batched=True,
    num_proc=8,
    batch_size=16,
    load_from_cache_file=False
)
column_names = infer_examples.column_names
infer_dataset = infer_examples.map(
    process_data_to_model_inputs,
    # do_train=False,
    batched=True,
    num_proc=8,
    batch_size=16,
    remove_columns=column_names,
    load_from_cache_file=False, # necessary
)

# set Python list to PyTorch tensor
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
infer_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# # enable fp16 apex training


model = AutoModelForSequenceClassification.from_pretrained(VER_MODEL_PATH)
model.resize_token_embeddings(len(tokenizer))

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=train_dataset
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    pred = trainer.predict(infer_dataset, ignore_keys=["encoder_last_hidden_state"])
    create_ver_infer(infer_examples, pred)



