import os
import random
import logging
import shutil
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer, XLMRobertaModel

Proj_Abs_Dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_data_path = os.getenv("AMLT_DATA_DIR", default=os.path.join(Proj_Abs_Dir, "data"))
_pretrained_models_path = os.path.join(_data_path, "pretrained_models")

def load_pretrained_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    tokenizer_path = os.path.join(_pretrained_models_path, tokenizer_name)
    if os.path.exists(tokenizer_path):
        logging.info('loading pretrained tokenizer from {} ...'.format(tokenizer_path))
        return AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        return AutoTokenizer.from_pretrained(tokenizer_name)

def save_pretrained_tokenizer(tokenizer_name: str, saved_dir: str):
    tokenizer_files = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model', 'added_tokens.json']
    tokenizer_path = os.path.join(_pretrained_models_path, tokenizer_name)
    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(saved_dir)
    else:
        os.makedirs(saved_dir, exist_ok=True)
        for tokenizer_file in tokenizer_files:
            logging.info("Copy {} to {} ...".format(tokenizer_file, saved_dir))
            shutil.copyfile(os.path.join(tokenizer_path, tokenizer_file), os.path.join(saved_dir, tokenizer_file))
    pass

def load_pretrained_model(model_name: str, torchscript=False) -> PreTrainedModel:
    model_path = os.path.join(_pretrained_models_path, model_name)
    if os.path.exists(model_path):
        logging.info('loading pretrained model from {} ...'.format(model_path))
        return AutoModel.from_pretrained(model_path, torchscript=torchscript)
    else:
        return AutoModel.from_pretrained(model_name, torchscript=False)

def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except:
        return False

def set_seed(seed = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def timer(func):
    def func_wrapper(*args,**kwargs):
        from datetime import datetime
        time_start = datetime.now()
        result = func(*args,**kwargs)
        cost = datetime.now() - time_start
        logging.info('Run {} over, cost: {}.'.format(func.__name__, str(cost)))
        return result
    return func_wrapper

class LambdaFunc:
    def __init__(self, func_str: str) -> None:
        self.func = self._parse_func_from_str(func_str)

    @staticmethod
    def _parse_func_from_str(func_str: str):
        items = func_str.split('_')
        if len(items) == 1 and is_float(func_str):
            weight = float(func_str)
            return lambda _: weight
        else:
            assert len(items) == 2 or len(items) == 3
            max_weight = 1.0 if len(items) == 2 else float(items[2])
            start_step, end_step = int(items[0]), int(items[1])
            assert end_step > start_step
            return lambda x: min(1.0, max(0.0, (x - start_step) / (end_step - start_step))) * max_weight

    def __call__(self, step: int) -> float:
        return self.func(step)

def fetch_items_from_dict(inputs: Dict, rule):
    outputs = {}
    for key, val in inputs.items():
        if isinstance(rule, str):
            if key == rule:
                outputs[key] = val
        elif isinstance(rule, List):
            if key in rule:
                outputs[key] = val
        elif callable(rule):
            if rule(key) == True:
                outputs[key] = val
        else:
            raise NotImplementedError(type(rule))
    return outputs