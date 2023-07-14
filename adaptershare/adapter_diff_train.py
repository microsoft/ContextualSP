# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import argparse
import json
import os
import random
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from pretrained_models import *
# from tensorboardX import SummaryWriter

# from torch.utils.tensorboard import SummaryWriter
from experiments.exp_def import TaskDefs
from mt_dnn.inference import eval_model, extract_encoding
from data_utils.log_wrapper import create_logger
from data_utils.task_def import EncoderModelType
from data_utils.utils import set_environment
from mt_dnn.batcher import (
    SingleTaskDataset,
    MultiTaskDataset,
    Collater,
    MultiTaskBatchSampler,
    DistMultiTaskBatchSampler,
    DistSingleTaskBatchSampler,
    TaskIterBatchSampler
)
from mt_dnn.batcher import DistTaskDataset
from mt_dnn.adapter_diff_model import MTDNNModel

from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    MultiLingAdapterArguments
)


@dataclass
class ModelArguments:
    update_bert_opt: int = field(
        default=0, metadata={"help": "BERT freeze or not"}
    )

    multi_gpu_on: bool = field(
        default=False, metadata={"help": "distributed training"}
    )

    mem_cum_type: str = field(
        default='simple', metadata={"help": "bilinear/simple/defualt"}
    )

    answer_num_turn: int = field(
        default=5, metadata={"help": "answer_num_turn"}
    )

    answer_mem_drop_p: float = field(
        default=0.1, metadata={"help": "answer_mem_drop_p"}
    )

    answer_att_hidden_size: int = field(
        default=128, metadata={"help": "answer_att_hidden_size"}
    )

    answer_att_type: str = field(
        default='bilinear', metadata={"help": "bilinear/simple/defualt"}
    )
    
    answer_rnn_type: str = field(
        default='gru', metadata={"help": "rnn/gru/lstm"}
    )
    
    answer_sum_att_type: str = field(
        default='bilinear', metadata={"help": "bilinear/simple/defualt"}
    )
    
    answer_merge_opt: int = field(
        default=1, metadata={"help": "answer_merge_opt"}
    )

    answer_mem_type: int = field(
        default=1, metadata={"help": "answer_mem_type"}
    )

    max_answer_len: int = field(
        default=10, metadata={"help": "max_answer_len"}
    )

    answer_dropout_p: float = field(
        default=0.1, metadata={"help": "answer_dropout_p"}
    )

    answer_weight_norm_on: bool = field(
        default=False, metadata={"help": "answer_weight_norm_on"}
    )

    dump_state_on: bool = field(
        default=False, metadata={"help": "dump_state_on"}
    )

    answer_opt: int = field(
        default=1, metadata={"help": "0,1"}
    )

    pooler_actf: str = field(
        default='tanh', metadata={"help": "tanh/relu/gelu"}
    )

    mtl_opt: int = field(
        default=0, metadata={"help": "mtl_opt"}
    )

    ratio: float = field(
        default=0, metadata={"help": "ratio"}
    )

    mix_opt: int = field(
        default=0, metadata={"help": "mix_opt"}
    )
    
    max_seq_len: int = field(
        default=512, metadata={"help": "max_seq_len"}
    )

    init_ratio: int = field(
        default=1, metadata={"help": "init_ratio"}
    )

    encoder_type: int = field(
        default=EncoderModelType.BERT, metadata={"help": "encoder_type"}
    )

    num_hidden_layers: int = field(
        default=-1, metadata={"help": "num_hidden_layers"}
    )

    # BERT pre-training
    bert_model_type: str = field(
        default='bert-base-uncased', metadata={"help": "bert_model_type"}
    )

    do_lower_case: bool = field(
        default=True, metadata={"help": "do_lower_case"}
    )

    masked_lm_prob: float = field(
        default=0.15, metadata={"help": "masked_lm_prob"}
    )

    short_seq_prob: float = field(
        default=0.2, metadata={"help": "short_seq_prob"}
    )

    max_predictions_per_seq: int = field(
        default=128, metadata={"help": "max_predictions_per_seq"}
    )

    # bin samples
    bin_on: bool = field(
        default=False, metadata={"help": "bin_on"}
    )

    bin_size: int = field(
        default=64, metadata={"help": "bin_size"}
    )

    bin_grow_ratio: float = field(
        default=0.5, metadata={"help": "bin_size"}
    )

    # dist training

    local_rank: int = field(
        default=-1, metadata={"help": "For distributed training: local_rank"}
    )

    world_size: int = field(
        default=1, metadata={"help": "For distributed training: world_size"}
    )

    master_addr: str = field(
        default='localhost', metadata={"help": "master_addr"}
    )

    master_port: str = field(
        default='6600', metadata={"help": "master_port"}
    )

    backend: str = field(
        default='nccl', metadata={"help": "backend"}
    )



@dataclass
class DataArguments:

    log_file: str = field(
        default='mt-dnn-train.log', metadata={"help": "path for log file."}
    )

    tensorboard: bool = field(
        default=False, metadata={"help": "tensorboard"}
    )
    
    tensorboard_logdir: str = field(
        default='tensorboard_logdir', metadata={"help": "tensorboard_logdir"}
    )

    data_dir: str = field(
        default='data/canonical_data/bert_uncased_lower', metadata={"help": "data_dir"}
    )

    data_sort_on: bool = field(
        default=False, metadata={"help": "data_sort_on"}
    )

    name: str = field(
        default='farmer', metadata={"help": "name"}
    )

    task_def: str = field(
        default='experiments/glue/glue_task_def.yml', metadata={"help": "task_def"}
    )

    train_datasets: str = field(
        default='mnli,mrpc', metadata={"help": "train_datasets"}
    )

    test_datasets: str = field(
        default='mnli_matched,mnli_mismatched', metadata={"help": "test_datasets"}
    )

    glue_format_on: bool = field(
        default=False, metadata={"help": "glue_format_on"}
    )
    
    mkd_opt: int = field(
        default=0, metadata={"help": ">0 to turn on knowledge distillation, requires 'softlabel' column in input data"}
    )

    do_padding: bool = field(
        default=False, metadata={"help": "do_padding"}
    )


@dataclass
class TrainingArguments:

    cuda: bool = field(
        default=torch.cuda.is_available(), metadata={"help": "whether to use GPU acceleration."}
    )

    init_checkpoint: str = field(
        default='bert-base-uncased', metadata={"help": "init_checkpoint"}
    )

    log_per_updates: int = field(
        default=500, metadata={"help": "log_per_updates"}
    )

    save_per_updates: int = field(
        default=10000, metadata={"help": "save_per_updates"}
    )

    save_per_updates_on: bool = field(
        default=True, metadata={"help": "save_per_updates_on"}
    )
    
    epochs: int = field(
        default=5, metadata={"help": "epochs"}
    )

    batch_size: int = field(
        default=8, metadata={"help": "batch_size"}
    )

    batch_size_eval: int = field(
        default=8, metadata={"help": "batch_size_eval"}
    )
    
    optimizer: str = field(
        default='adamax', metadata={"help": "supported optimizer: adamax, sgd, adadelta, adam"}
    )
    
    grad_clipping: float = field(
        default=0, metadata={"help": "grad_clipping"}
    )

    global_grad_clipping: float = field(
        default=1.0, metadata={"help": "global_grad_clipping"}
    )

    weight_decay: float = field(
        default=0, metadata={"help": "weight_decay"}
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "learning_rate"}
    )

    momentum: float = field(
        default=0, metadata={"help": "momentum"}
    )

    warmup: float = field(
        default=0.1, metadata={"help": "warmup"}
    )

    warmup_schedule: str = field(
        default='warmup_linear', metadata={"help": "warmup_schedule"}
    )

    adam_eps: float = field(
        default=1e-6, metadata={"help": "adam_eps"}
    )

    vb_dropout: bool = field(
        default=False, metadata={"help": "vb_dropout"}
    )

    dropout_p: float = field(
        default=0.1, metadata={"help": "dropout_p"}
    )

    dropout_w: float = field(
        default=0.000, metadata={"help": "dropout_w"}
    )

    bert_dropout_p: float = field(
        default=0.1, metadata={"help": "bert_dropout_p"}
    )

    # loading
    model_ckpt: str = field(
        default='checkpoints/model_0.pt', metadata={"help": "model_ckpt"}
    )

    resume: bool = field(
        default=False, metadata={"help": "resume"}
    )
    # scheduler
    scheduler_type: int = field(
        default=0, metadata={"help": "0: linear, 1: cosine, 2 constant"}
    )

    output_dir: str = field(
        default='checkpoint', metadata={"help": "output_dir"}
    )

    seed: int = field(
        default=2018, metadata={"help": "random seed for data shuffling, embedding init, etc."}
    )

    grad_accumulation_step: int = field(
        default=21018, metadata={"help": "grad_accumulation_step"}
    )

    ite_batch_num: int = field(
        default=500, metadata={"help": "ite_batch_num"}
    )
    
    # fp 16

    fp16: bool = field(
        default=False, metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"}
    )
    
    fp16_opt_level: str = field(
        default='O1', metadata={"help": "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html"}
    )

    # adv training
    adv_train: bool = field(
        default=False, metadata={"help": "adv_train"}
    )
    # the current release only includes smart perturbation
    adv_opt: int = field(
        default=0, metadata={"help": "adv_opt"}
    )

    adv_norm_level: int = field(
        default=0, metadata={"help": "adv_norm_level"}
    )

    adv_p_norm: str = field(
        default='inf', metadata={"help": "adv_p_norm"}
    )
    
    adv_alpha: float = field(
        default=1, metadata={"help": "adv_alpha"}
    )

    adv_k: int = field(
        default=1, metadata={"help": "adv_k"}
    )

    adv_step_size: float = field(
        default=1e-5, metadata={"help": "adv_step_size"}
    )

    adv_noise_var: float = field(
        default=1e-5, metadata={"help": "adv_noise_var"}
    )
    
    adv_epsilon: float = field(
        default=1e-6, metadata={"help": "adv_epsilon"}
    )

    encode_mode: bool = field(
        default=False, metadata={"help": "only encode test data"}
    )

    debug: bool = field(
        default=False, metadata={"help": "print debug info"}
    )

    # transformer cache
    transformer_cache: str = field(
        default='.cache', metadata={"help": "transformer_cache"}
    )


@dataclass
class DUMTLArguments(MultiLingAdapterArguments):
    train_adapter_fusion: bool = field(
        default=False, metadata={"help": "Train an adapter fusion for target task."}
    )

    finetune: bool = field(
        default=False, metadata={"help": "Finetuning on target task."}
    )

    load_adapter_fusion: Optional[str] = field(
        default="", metadata={"help": "Pre-trained adapter fusion module to be loaded from Hub."}
    )

    diff_structure_init: str = field(
        default=None,
        metadata={"help": "The initial differentitated adapter path."},
    )

    adapter_diff: bool = field(
        default=True,
        metadata={"help": "Differentitated adapter training mode."},
    )

    adapter_cache_path: str = field(
        default='checkpoint',
        metadata={"help": "The initial differentitated adapter path."},
    )

    min_intra_simiarity: int = field(
        default=2, metadata={"help": "math.cos(math.pi / min_intra_simiarity)"}
    )

    max_entropy_threshold: int = field(
        default=3, metadata={"help": "math.cos(math.pi / max_entropy_threshold)"}
    )

    max_interference_degree: float = field(
        default=0.5, metadata={"help": "max_interference_degree"}
    )

hfparser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DUMTLArguments)
    )
model_args, data_args, training_args, adapter_args = hfparser.parse_args_into_dataclasses()


output_dir = training_args.output_dir
data_dir = data_args.data_dir
data_args.train_datasets = data_args.train_datasets.split(",")
data_args.test_datasets = data_args.test_datasets.split(",")

os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

set_environment(training_args.seed, training_args.cuda)
log_path = data_args.log_file
logger = create_logger(__name__, to_disk=True, log_file=log_path)

task_defs = TaskDefs(data_args.task_def)
encoder_type = model_args.encoder_type


def dump(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def evaluation(
    model,
    datasets,
    data_list,
    task_defs,
    output_dir="checkpoints",
    epoch=0,
    n_updates=-1,
    with_label=False,
    tensorboard=None,
    glue_format_on=False,
    test_on=False,
    device=None,
    logger=None,
):
    # eval on rank 1
    print_message(logger, "Evaluation")
    test_prefix = "Test" if test_on else "Dev"
    if n_updates > 0:
        updates_str = "updates"
    else:
        updates_str = "epoch"
    updates = model.updates if n_updates > 0 else epoch
    for idx, dataset in enumerate(datasets):
        prefix = dataset.split("_")[0]

        model._switch_model_task_mode(prefix)

        task_def = task_defs.get_task_def(prefix)
        label_dict = task_def.label_vocab
        test_data = data_list[idx]
        if test_data is not None:
            with torch.no_grad():
                (
                    test_metrics,
                    test_predictions,
                    test_scores,
                    test_golds,
                    test_ids,
                ) = eval_model(
                    model,
                    test_data,
                    metric_meta=task_def.metric_meta,
                    device=device,
                    with_label=with_label,
                    label_mapper=label_dict,
                    task_type=task_def.task_type,
                )
            for key, val in test_metrics.items():
                if tensorboard:
                    tensorboard.add_scalar(
                        "{}/{}/{}".format(test_prefix, dataset, key),
                        val,
                        global_step=updates,
                    )
                if isinstance(val, str):
                    print_message(
                        logger,
                        "Task {0} -- {1} {2} -- {3} {4}: {5}".format(
                            dataset, updates_str, updates, test_prefix, key, val
                        ),
                        level=1,
                    )
                elif isinstance(val, float):
                    print_message(
                        logger,
                        "Task {0} -- {1} {2} -- {3} {4}: {5:.3f}".format(
                            dataset, updates_str, updates, test_prefix, key, val
                        ),
                        level=1,
                    )
                else:
                    test_metrics[key] = str(val)
                    print_message(
                        logger,
                        "Task {0} -- {1} {2} -- {3} {4}: \n{5}".format(
                            dataset, updates_str, updates, test_prefix, key, val
                        ),
                        level=1,
                    )

            if model_args.local_rank in [-1, 0]:
                score_file = os.path.join(
                    output_dir,
                    "{}_{}_scores_{}_{}.json".format(
                        dataset, test_prefix.lower(), updates_str, updates
                    ),
                )
                results = {
                    "metrics": test_metrics,
                    "predictions": test_predictions,
                    "uids": test_ids,
                    "scores": test_scores,
                }
                dump(score_file, results)
                if glue_format_on:
                    from experiments.glue.glue_utils import submit

                    official_score_file = os.path.join(
                        output_dir,
                        "{}_{}_scores_{}.tsv".format(
                            dataset, test_prefix.lower(), updates_str
                        ),
                    )
                    submit(official_score_file, results, label_dict)


def initialize_distributed(logger):
    """Initialize torch.distributed."""
    model_args.rank = int(os.getenv("RANK", "0"))
    model_args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    batch_size_pre_gpu = int(training_args.batch_size / model_args.world_size)
    print_message(logger, "Batch Size Per GPU: {}".format(batch_size_pre_gpu))

    device = model_args.rank % torch.cuda.device_count()
    if model_args.local_rank is not None:
        device = model_args.local_rank
    torch.cuda.set_device(device)
    device = torch.device("cuda", model_args.local_rank)
    # Call the init process
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6600")
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend=model_args.backend,
        world_size=model_args.world_size,
        rank=model_args.rank,
        init_method=init_method,
    )
    return device


def print_message(logger, message, level=0):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            do_logging = True
        else:
            do_logging = False
    else:
        do_logging = True
    if do_logging:
        if level == 1:
            logger.warning(message)
        else:
            logger.info(message)


def main():
    # set up dist
    device = torch.device("cuda")
    if model_args.local_rank > -1:
        device = initialize_distributed(logger)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    opt = [(k, eval(f'training_args.{k}')) for k in vars(training_args)]
    opt.extend([(k, eval(f'model_args.{k}')) for k in vars(model_args)])
    opt.extend([(k, eval(f'data_args.{k}')) for k in vars(data_args)])
    opt.extend([(k, eval(f'adapter_args.{k}')) for k in vars(adapter_args)])
    opt = dict(opt)

    # update data dir
    opt["data_dir"] = data_dir
    batch_size = training_args.batch_size
    print_message(logger, "Launching the MT-DNN training")
    # return
    tasks = {}
    task_def_list = []
    dropout_list = []
    printable = model_args.local_rank in [-1, 0]

    train_datasets = []
    for dataset in data_args.train_datasets:
        prefix = dataset.split("_")[0]
        if prefix in tasks:
            continue
        task_id = len(tasks)
        tasks[prefix] = task_id
        task_def = task_defs.get_task_def(prefix)
        task_def_list.append(task_def)
        train_path = os.path.join(data_dir, "{}_train.json".format(dataset))
        print_message(logger, "Loading {} as task {}".format(train_path, task_id))
        train_data_set = SingleTaskDataset(
            train_path,
            True,
            maxlen=model_args.max_seq_len,
            task_id=task_id,
            task_def=task_def,
            printable=printable,
        )
        train_datasets.append(train_data_set)
    train_collater = Collater(
        dropout_w=training_args.dropout_w,
        encoder_type=encoder_type,
        soft_label=data_args.mkd_opt > 0,
        max_seq_len=model_args.max_seq_len,
        do_padding=data_args.do_padding,
    )
    multi_task_train_dataset = MultiTaskDataset(train_datasets)
    if model_args.local_rank != -1:
        multi_task_batch_sampler = DistMultiTaskBatchSampler(
            train_datasets,
            training_args.batch_size,
            model_args.mix_opt,
            model_args.ratio,
            rank=model_args.local_rank,
            world_size=model_args.world_size,
        )
    else:
        multi_task_batch_sampler = TaskIterBatchSampler(
            train_datasets,
            training_args.batch_size,
            model_args.mix_opt,
            model_args.ratio,
            bin_on=model_args.bin_on,
            bin_size=model_args.bin_size,
            bin_grow_ratio=model_args.bin_grow_ratio,
            ite_batch_num=training_args.ite_batch_num
        )
    multi_task_train_data = DataLoader(
        multi_task_train_dataset,
        batch_sampler=multi_task_batch_sampler,
        collate_fn=train_collater.collate_fn,
        pin_memory=training_args.cuda,
    )

    id_task_map = dict([(v, k) for k, v in tasks.items()])

    opt["task_def_list"] = task_def_list

    dev_data_list = []
    test_data_list = []
    heldout_eval_data_list = []
    test_collater = Collater(
        is_train=False,
        encoder_type=encoder_type,
        max_seq_len=model_args.max_seq_len,
        do_padding=data_args.do_padding,
    )
    for dataset in data_args.test_datasets:
        prefix = dataset.split("_")[0]
        task_def = task_defs.get_task_def(prefix)
        task_id = tasks[prefix]
        task_type = task_def.task_type
        data_type = task_def.data_type

        dev_path = os.path.join(data_dir, "{}_dev.json".format(dataset))
        dev_data = None
        if os.path.exists(dev_path):
            dev_data_set = SingleTaskDataset(
                dev_path,
                False,
                maxlen=model_args.max_seq_len,
                task_id=task_id,
                task_def=task_def,
                printable=printable,
            )
            if model_args.local_rank != -1:
                dev_data_set = DistTaskDataset(dev_data_set, task_id)
                single_task_batch_sampler = DistSingleTaskBatchSampler(
                    dev_data_set,
                    training_args.batch_size_eval,
                    rank=model_args.local_rank,
                    world_size=model_args.world_size,
                )
                dev_data = DataLoader(
                    dev_data_set,
                    batch_sampler=single_task_batch_sampler,
                    collate_fn=test_collater.collate_fn,
                    pin_memory=training_args.cuda,
                )
            else:
                dev_data = DataLoader(
                    dev_data_set,
                    batch_size=training_args.batch_size_eval,
                    collate_fn=test_collater.collate_fn,
                    pin_memory=training_args.cuda,
                )
        dev_data_list.append(dev_data)

        tmp_heldout_eval_data_list = []
        if os.path.exists(dev_path):
            for hs in [0, 10]:
                dev_data_set = SingleTaskDataset(
                    dev_path,
                    True,
                    maxlen=model_args.max_seq_len,
                    task_id=task_id,
                    task_def=task_def,
                    printable=printable,
                    heldout_start=hs
                )
                # if model_args.local_rank != -1:
                #     dev_data_set = DistTaskDataset(dev_data_set, task_id)
                #     single_task_batch_sampler = DistSingleTaskBatchSampler(
                #         dev_data_set,
                #         training_args.batch_size_eval,
                #         rank=model_args.local_rank,
                #         world_size=model_args.world_size,
                #     )
                #     heldout_eval_data = DataLoader(
                #         dev_data_set,
                #         batch_sampler=single_task_batch_sampler,
                #         collate_fn=test_collater.collate_fn,
                #         pin_memory=training_args.cuda,
                #     )
                # else:
                #     heldout_eval_data = DataLoader(
                #         dev_data_set,
                #         batch_size=training_args.batch_size_eval,
                #         collate_fn=test_collater.collate_fn,
                #         pin_memory=training_args.cuda,
                #     )
                tmp_heldout_eval_data_list.append(dev_data_set)
        heldout_eval_data_list.append(tmp_heldout_eval_data_list)

        test_path = os.path.join(data_dir, "{}_test.json".format(dataset))
        test_data = None
        if os.path.exists(test_path):
            test_data_set = SingleTaskDataset(
                test_path,
                False,
                maxlen=model_args.max_seq_len,
                task_id=task_id,
                task_def=task_def,
                printable=printable,
            )
            if model_args.local_rank != -1:
                test_data_set = DistTaskDataset(test_data_set, task_id)
                single_task_batch_sampler = DistSingleTaskBatchSampler(
                    test_data_set,
                    training_args.batch_size_eval,
                    rank=model_args.local_rank,
                    world_size=model_args.world_size,
                )
                test_data = DataLoader(
                    test_data_set,
                    batch_sampler=single_task_batch_sampler,
                    collate_fn=test_collater.collate_fn,
                    pin_memory=training_args.cuda,
                )
            else:
                test_data = DataLoader(
                    test_data_set,
                    batch_size=training_args.batch_size_eval,
                    collate_fn=test_collater.collate_fn,
                    pin_memory=training_args.cuda,
                )
        test_data_list.append(test_data)
    
    heldout_eval_data_list = [[hs[0] for hs in heldout_eval_data_list], [hs[1] for hs in heldout_eval_data_list]]
    heldout_eval_data_list1 = MultiTaskDataset(heldout_eval_data_list[0])
    heldout_eval_data_list2 = MultiTaskDataset(heldout_eval_data_list[1])
    heldout_eval_dataset_list = [heldout_eval_data_list1, heldout_eval_data_list2]
    tmp_heldout_eval_data_list = []
    # TODO
    for hi, heldout_datasets in enumerate(heldout_eval_data_list):
        multi_task_batch_sampler = MultiTaskBatchSampler(
            heldout_datasets,
            training_args.batch_size,
            model_args.mix_opt,
            model_args.ratio,
            bin_on=model_args.bin_on,
            bin_size=model_args.bin_size,
            bin_grow_ratio=model_args.bin_grow_ratio,
            heldout=True
        )

        multi_task_heldout_data = DataLoader(
            heldout_eval_dataset_list[hi],
            batch_sampler=multi_task_batch_sampler,
            collate_fn=train_collater.collate_fn,
            pin_memory=training_args.cuda,
        )

        tmp_heldout_eval_data_list.append(multi_task_heldout_data)
    
    heldout_eval_data_list = tmp_heldout_eval_data_list

    # for data_loader in heldout_eval_data_list1:
    #     for (batch_meta, batch_data) in data_loader:
    #         batch_meta, batch_data = Collater.patch_data(device, batch_meta, batch_data)
    #         print(id_task_map[batch_meta["task_id"]])
    #         print(len(batch_data))
    #         exit(0)

    print_message(logger, "#" * 20)
    print_message(logger, opt)
    print_message(logger, "#" * 20)

    # div number of grad accumulation.
    num_all_batches = (
        training_args.epochs * len(multi_task_train_data) // training_args.grad_accumulation_step
    )
    print_message(logger, "############# Gradient Accumulation Info #############")
    print_message(
        logger, "number of step: {}".format(training_args.epochs * len(multi_task_train_data))
    )
    print_message(
        logger,
        "number of grad grad_accumulation step: {}".format(training_args.grad_accumulation_step),
    )
    print_message(logger, "adjusted number of step: {}".format(num_all_batches))
    print_message(logger, "############# Gradient Accumulation Info #############")

    init_model = training_args.init_checkpoint
    state_dict = None

    if os.path.exists(init_model):
        if (
            encoder_type == EncoderModelType.BERT
            or encoder_type == EncoderModelType.DEBERTA
            or encoder_type == EncoderModelType.ELECTRA
        ):
            state_dict = torch.load(init_model, map_location=device)
            config = state_dict["config"]
        elif (
            encoder_type == EncoderModelType.ROBERTA
            or encoder_type == EncoderModelType.XLM
        ):
            model_path = "{}/model.pt".format(init_model)
            state_dict = torch.load(model_path, map_location=device)
            arch = state_dict["args"].arch
            arch = arch.replace("_", "-")
            if encoder_type == EncoderModelType.XLM:
                arch = "xlm-{}".format(arch)
            # convert model arch
            from data_utils.roberta_utils import update_roberta_keys
            from data_utils.roberta_utils import patch_name_dict

            state = update_roberta_keys(
                state_dict["model"], nlayer=state_dict["args"].encoder_layers
            )
            state = patch_name_dict(state)
            literal_encoder_type = EncoderModelType(opt["encoder_type"]).name.lower()
            config_class, model_class, tokenizer_class = MODEL_CLASSES[
                literal_encoder_type
            ]
            config = config_class.from_pretrained(arch).to_dict()
            state_dict = {"state": state}
    else:
        if opt["encoder_type"] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        literal_encoder_type = EncoderModelType(opt["encoder_type"]).name.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
        config = config_class.from_pretrained(
            init_model, cache_dir=training_args.transformer_cache
        ).to_dict()

    config["attention_probs_dropout_prob"] = training_args.bert_dropout_p
    config["hidden_dropout_prob"] = training_args.bert_dropout_p
    config["multi_gpu_on"] = model_args.multi_gpu_on
    if model_args.num_hidden_layers > 0:
        config["num_hidden_layers"] = model_args.num_hidden_layers

    # opt.update(config)

    model = MTDNNModel(
        opt, 
        device=device, 
        state_dict=state_dict, 
        num_train_step=num_all_batches, 
        adapter_args=adapter_args, 
        adapter=True, 
        task_name='-'.join(list(tasks.keys())), 
        id_task_map=id_task_map,
        heldout_eval_dataset=heldout_eval_data_list
    )
    if training_args.resume and training_args.model_ckpt:
        print_message(logger, "loading model from {}".format(training_args.model_ckpt))
        model.load(training_args.model_ckpt)

    #### model meta str
    headline = "############# Model Arch of MT-DNN #############"
    ### print network
    print_message(logger, "\n{}\n{}\n".format(headline, model.network))

    # dump config
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "w", encoding="utf-8") as writer:
        writer.write("{}\n".format(json.dumps(opt)))
        writer.write("\n{}\n{}\n".format(headline, model.network))

    print_message(logger, "Total number of params: {}".format(model.total_param))

    # tensorboard
    tensorboard = None
    # if args.tensorboard:
    #     args.tensorboard_logdir = os.path.join(args.output_dir, args.tensorboard_logdir)
    #     tensorboard = SummaryWriter(log_dir=args.tensorboard_logdir)

    if training_args.encode_mode:
        for idx, dataset in enumerate(data_args.test_datasets):
            prefix = dataset.split("_")[0]
            test_data = test_data_list[idx]
            with torch.no_grad():
                encoding = extract_encoding(model, test_data, use_cuda=training_args.cuda)
            torch.save(
                encoding, os.path.join(output_dir, "{}_encoding.pt".format(dataset))
            )
        return

    training_args.ite_batch_num = 5
    diff_operation = True
    differentiate_detect_step = training_args.ite_batch_num * len(id_task_map)
    differentiate_start_step = 0
    differentiate_rate_threshold = len(id_task_map)
    for epoch in range(0, training_args.epochs):
        print_message(logger, "At epoch {}".format(epoch), level=1)
        start = datetime.now()

        for i, (batch_meta, batch_data) in enumerate(multi_task_train_data):
            batch_meta, batch_data = Collater.patch_data(device, batch_meta, batch_data)
            task_id = batch_meta["task_id"]

            if id_task_map[task_id] != model.current_task:
                model._switch_model_task_mode(id_task_map[task_id])
                print(f'>>> Switch to {model.current_task} Task Successed !!!')

            model.update(batch_meta, batch_data)

            if (model.updates) % (training_args.log_per_updates) == 0 or model.updates == 1:
                ramaining_time = str(
                    (datetime.now() - start)
                    / (i + 1)
                    * (len(multi_task_train_data) - i - 1)
                ).split(".")[0]
                if training_args.adv_train and training_args.debug:
                    debug_info = " adv loss[%.5f] emb val[%.8f] eff_perturb[%.8f] " % (
                        model.adv_loss.avg,
                        model.emb_val.avg,
                        model.eff_perturb.avg,
                    )
                else:
                    debug_info = " "
                print_message(
                    logger,
                    "Task [{0:2}] updates[{1:6}] train loss[{2:.5f}]{3}remaining[{4}]".format(
                        task_id,
                        model.updates,
                        model.train_loss.avg,
                        debug_info,
                        ramaining_time,
                    ),
                )
                if data_args.tensorboard:
                    tensorboard.add_scalar(
                        "train/loss", model.train_loss.avg, global_step=model.updates
                    )

            # Differentiation Operation
            if model.updates > differentiate_start_step and model.updates % differentiate_detect_step == 0 and diff_operation:
                model._differentiate_operate()
                print(f'>>> Differentiation Operation Successed !!!')
                current_diff_rate = model._calculate_differentiated_rate()
                print(f'>>> current_diff_rate: {current_diff_rate}')
                if current_diff_rate >= differentiate_rate_threshold:
                        diff_operation = False
                model._switch_model_task_mode(model.current_task)
                # exit(0)

            if (
                training_args.save_per_updates_on
                and (
                    (model.local_updates)
                    % (training_args.save_per_updates * training_args.grad_accumulation_step)
                    == 0
                )
                and model_args.local_rank in [-1, 0]
            ):
                ckpt_dir = f'checkpoint-{epoch}-{model.updates}'
                ckpt_dir = os.path.join(output_dir, ckpt_dir)

                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                model_file = os.path.join(
                    ckpt_dir, "model_{}_{}.pt".format(epoch, model.updates)
                )
                evaluation(
                    model,
                    data_args.test_datasets,
                    dev_data_list,
                    task_defs,
                    ckpt_dir,
                    epoch,
                    n_updates=training_args.save_per_updates,
                    with_label=True,
                    tensorboard=tensorboard,
                    glue_format_on=data_args.glue_format_on,
                    test_on=False,
                    device=device,
                    logger=logger,
                )
                evaluation(
                    model,
                    data_args.test_datasets,
                    test_data_list,
                    task_defs,
                    ckpt_dir,
                    epoch,
                    n_updates=training_args.save_per_updates,
                    with_label=False,
                    tensorboard=tensorboard,
                    glue_format_on=data_args.glue_format_on,
                    test_on=True,
                    device=device,
                    logger=logger,
                )
                print_message(logger, "Saving mt-dnn model to {}".format(model_file))
                model.save(model_file)

        ckpt_dir = f'checkpoint-{epoch}'
        ckpt_dir = os.path.join(output_dir, ckpt_dir)

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        evaluation(
            model,
            data_args.test_datasets,
            dev_data_list,
            task_defs,
            ckpt_dir,
            epoch,
            with_label=True,
            tensorboard=tensorboard,
            glue_format_on=data_args.glue_format_on,
            test_on=False,
            device=device,
            logger=logger,
        )
        evaluation(
            model,
            data_args.test_datasets,
            test_data_list,
            task_defs,
            ckpt_dir,
            epoch,
            with_label=False,
            tensorboard=tensorboard,
            glue_format_on=data_args.glue_format_on,
            test_on=True,
            device=device,
            logger=logger,
        )
        print_message(logger, "[new test scores at {} saved.]".format(epoch))
        if model_args.local_rank in [-1, 0]:
            model_file = os.path.join(ckpt_dir, "model_{}.pt".format(epoch))
            model.save(model_file)
    if data_args.tensorboard:
        tensorboard.close()


if __name__ == "__main__":
    main()