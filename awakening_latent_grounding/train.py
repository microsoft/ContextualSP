import abc
import os
import json
import argparse
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import multiprocessing as mp

# import wandb
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from transformers import AdamW, get_linear_schedule_with_warmup

from contracts import *
from models import *
from utils import *


@dataclass
class TrainingArgs:
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for Adam."})
    pretrained_learning_rate: float = field(default=None)
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_warmup_steps: int = field(default=2000)

    use_amp: bool = field(default=False)
    batch_size: int = field(default=1)
    max_train_steps: int = field(default=10, metadata={"help": "Training steps"})
    max_encode_length: int = field(default=512)
    logging_steps: int = field(default=1)
    evaluate_steps: int = field(default=1000)
    num_threads: int = field(default=None)
    grad_accumulation_steps: int = field(default=1)
    grounding_loss_weight_func: str = field(default='0.0')

    model: str = field(default='UniGrounding')
    pretrained_model: str = field(default="bert-base-uncased", metadata={"help": "Training epochs"})
    hidden_size: int = field(default=384)
    checkpoint: str = field(default=None)
    dropout: float = field(default=0.3)
    neg_penalty_weight: float = field(default=0.05)
    label_smoothing: float = field(default=0.0)
    remove_concept_dependency: bool = field(default=False)

    data_augment: bool = field(default=False)
    ignore_unmatched_values: bool = field(default=True)

    experiment_id: str = field(default=None)
    data_dir: str = field(default="data", metadata={"help": "input data dir"})
    datasets: str = field(default=None)
    output_dir: str = field(default='out', metadata={'help': 'output data dir'})
    sampling: bool = field(default=False)
    use_wandb: bool = field(default=False)
    device: str = field(default='cpu')
    seed: int = field(default=123)


def get_logger(path: str):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s\t%(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=path,
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s\t%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logger = logging.getLogger("UniGrounding")
    logger.addHandler(console)
    return logger


class Trainer:
    args: TrainingArgs
    model: nn.Module
    logger: logging.Logger
    experiment_name: str

    def __init__(self, args: TrainingArgs) -> None:
        self.args = args
        # set_seed(seed=self.args.seed)
        # deepspeed_plugin = DeepSpeedPlugin(zero_stage=args.grad_accumulation_steps, gradient_accumulation_steps=args.grad_accumulation_steps)
        self.accelerator = Accelerator(
            # fp16=self.args.use_amp,
            # deepspeed_plugin=deepspeed_plugin,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        self.experiment_name = "{}_G{}".format(args.model, args.pretrained_model)

        if self.args.sampling:
            self.experiment_name = "Sampling_" + self.experiment_name
            self.args.evaluate_steps = 100

        os.makedirs(os.path.join(self.args.output_dir, self.experiment_name), exist_ok=True)
        if self.accelerator.is_local_main_process:
            self.logger = get_logger(
                os.path.join(self.args.output_dir, self.experiment_name, '{}.log'.format(self.args.model)))
            self.logger.info(self.accelerator.state.__repr__())

        raw_model = self.load_model()
        # self.info("Model structure: {}\n".format(raw_model))
        self.tokenizer = raw_model.tokenizer
        self.model = self.accelerator.prepare(raw_model)
        self.generator = ETAGroundingGenerator(self.model, tokenizer=self.tokenizer,
                                               cp_batch_size=self.args.batch_size * 2,
                                               neg_penalty_weight=self.args.neg_penalty_weight)

        if self.accelerator.is_local_main_process:
            with open(os.path.join(self.args.output_dir, self.experiment_name, "config.json"), 'w',
                      encoding='utf-8') as fw:
                json.dump(self.args.__dict__, fw, indent=4, sort_keys=True)
                self.info("Save training config over.")
                for key, val in self.args.__dict__.items():
                    self.info("{} = {}".format(key, val))

    def load_model(self) -> BaseGroundingModel:
        config = {
            'model': self.args.model,
            'pretrained_model': self.args.pretrained_model,
            'hidden_size': self.args.hidden_size,
            'dropout': self.args.dropout,
            'num_keywords': len(All_Agg_Op_Keywords),
            'num_labels': len(All_Question_Labels),
            'label_smoothing': self.args.label_smoothing
        }

        model = load_grounding_model(self.args.model, config=config, checkpoint=self.args.checkpoint)
        self.info("Save model config ...")
        if self.accelerator.is_local_main_process:
            with open(os.path.join(self.args.output_dir, self.experiment_name, "model_config.json"), 'w',
                      encoding='utf-8') as fw:
                json.dump(config, fw, indent=4, sort_keys=True)

        return model

    def save(self, saved_path: str):
        if self.accelerator.is_local_main_process:
            model_weights = self.model.state_dict()
            if isinstance(self.model, DistributedDataParallel):
                model_weights = self.model.module.state_dict()
            elif isinstance(self.model, nn.Module):
                model_weights = self.model.state_dict()
            else:
                raise NotImplementedError(type(self.model))

            torch.save(model_weights, saved_path)
            self.info("Save checkpoint to {}".format(saved_path))

    def get_data_loader(self, split_name: str) -> DataLoader:
        assert split_name in ['train', 'dev', 'test']
        is_training = split_name == 'train'

        all_examples = []
        for source in self.args.datasets.split(','):
            data_path = os.path.join(self.args.data_dir, source.strip(), "{}.preproc.json".format(split_name))
            raw_examples: List[Text2SQLExample] = load_json_objects(Text2SQLExample, data_path)
            raw_count = len(raw_examples)
            raw_examples = [ex for ex in raw_examples if ex.resolved]
            self.info("Ignore {}-{} unresolved examples over, ignored size = {}.".format(
                split_name, source, raw_count - len(raw_examples)
            ))

            if self.args.sampling:
                sampling_size = 1 * self.args.batch_size
                # raw_examples = random.sample(raw_examples, sampling_size)
                raw_examples = raw_examples[:sampling_size]
                self.info(
                    "Sample {}-{} examples over, {} => {}.".format(split_name, source, raw_count, len(raw_examples)))

            self.info("Load {}-{} {} examples over.".format(split_name, source, len(raw_examples)))
            all_examples += raw_examples

        if self.args.ignore_unmatched_values:
            self.info("Ignore unmatched values for each example.")
            all_examples = [x.ignore_unmatched_values() for x in all_examples]

        data_iter = load_data_loader(
            examples=all_examples,
            tokenizer=self.tokenizer,
            batch_size=self.args.batch_size,
            max_enc_length=self.args.max_encode_length if is_training else self.tokenizer.model_max_length,
            is_training=is_training,
            n_processes=self.args.num_threads if self.args.num_threads else min(mp.cpu_count(), 16)
        )

        self.info("Load {} data loader over, batches = {}, examples = {}.".format(split_name, len(data_iter),
                                                                                  len(data_iter.dataset)))
        return data_iter

    def get_train_and_eval_iter(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_iter = self.get_data_loader('train')
        dev_iter = self.get_data_loader('dev')
        test_iter = self.get_data_loader('test')
        return train_iter, dev_iter, test_iter

    def info(self, msg: str):
        if self.accelerator.is_local_main_process:
            self.logger.info(msg)

    def log_metrics(self, metrics: Dict[str, float], prefix=None):
        pass

    @abc.abstractmethod
    def evaluate(self, dev_iter: DataLoader, saved_path: str = None, grounding_loss_weight: float = 0.0) -> Dict[
        str, float]:
        self.model.eval()
        evaluator = GroundingEvaluator(dataset=dev_iter.dataset)
        with torch.no_grad():
            for batch_inputs in dev_iter:
                batch_inputs = to_device(batch_inputs, self.accelerator.device)
                if grounding_loss_weight > 1e-3:
                    pseudo_grounding_labels = self.generator.generate(inputs=batch_inputs, is_training=False)
                    batch_inputs['grounding_labels'] = pseudo_grounding_labels
                    batch_inputs['grounding_loss_weight'] = grounding_loss_weight

                # todo: if from evaluation, question label should be gold (as well as loss computing)
                outputs = self.model(**batch_inputs)
                evaluator.add_batch(batch_inputs, outputs)

        self.model.train()
        return evaluator.get_metrics(log_saved_file=saved_path)

    def train(self):
        train_iter, dev_iter, test_iter = self.get_train_and_eval_iter()
        num_training_steps = self.args.max_train_steps * self.args.grad_accumulation_steps
        self.info("Number of training steps = {}".format(num_training_steps))

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if n.startswith("encoder")],
             'lr': self.args.pretrained_learning_rate},
            {'params': [p for n, p in self.model.named_parameters() if not (n.startswith("encoder"))],
             'lr': self.args.learning_rate}
        ]

        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        optimizer, train_iter = self.accelerator.prepare(optimizer, train_iter)

        # Dynamic update grounding loss weight
        glw_func = LambdaFunc(self.args.grounding_loss_weight_func)

        self.model.train()
        global_step, epoch = 0, 0
        logging_loss = defaultdict(float)

        while global_step < num_training_steps:
            epoch += 1
            for batch_inputs in train_iter:
                global_step += 1

                if global_step > num_training_steps:
                    break

                # Forward and compute loss
                grounding_loss_weight = glw_func(global_step)
                # self.log_metrics({ 'grounding_loss_weight' : grounding_loss_weight })
                if grounding_loss_weight > 1e-3:
                    pseudo_grounding_labels = self.generator.generate(inputs=batch_inputs)
                    batch_inputs['grounding_labels'] = pseudo_grounding_labels
                    batch_inputs['grounding_loss_weight'] = grounding_loss_weight

                outputs = self.model(**batch_inputs)
                loss = outputs['loss'] / self.args.grad_accumulation_steps
                self.accelerator.backward(loss)

                for key, val in fetch_items_from_dict(outputs, lambda x: x.endswith('loss')).items():
                    logging_loss[key] += val.item()
                    self.log_metrics({key: val.item()}, prefix='Train')

                if global_step % self.args.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if global_step % self.args.logging_steps == 0:
                    loss_string = " ".join(["Avg {}: {:.4f};".format(key, val / self.args.logging_steps) for key, val in
                                            logging_loss.items()])
                    self.info("Epoch: {:>3}; Step: {:>5}/{}; {}; Grounding Loss Weight: {}".format(epoch, global_step,
                                                                                                   num_training_steps,
                                                                                                   loss_string,
                                                                                                   grounding_loss_weight))

                    logging_loss = defaultdict(float)

                if global_step % self.args.evaluate_steps == 0:
                    self.info("Evaluating step {} ...".format(global_step))

                    dev_saved_path = os.path.join(self.args.output_dir, self.experiment_name,
                                                  '{}.step_{}.dev.txt'.format(self.args.model, global_step))
                    dev_metrics = self.evaluate(dev_iter, saved_path=dev_saved_path,
                                                grounding_loss_weight=grounding_loss_weight)
                    self.info("Evaluate Dev over:\n{}\n".format("\n".join(
                        [f"Dev {k} = {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in
                         dev_metrics.items()])))
                    self.log_metrics({key: val for key, val in dev_metrics.items() if isinstance(val, float)},
                                     prefix='Dev')
                    # self.log_metrics({ key : val for key, val in dev_metrics.items() if isinstance(val, float) and not ("wikisql" in key or "annatalk" in key) }, prefix='Dev')

                    test_saved_path = os.path.join(self.args.output_dir, self.experiment_name,
                                                   '{}.step_{}.test.txt'.format(self.args.model, global_step))
                    test_metrics = self.evaluate(test_iter, saved_path=test_saved_path,
                                                 grounding_loss_weight=grounding_loss_weight)
                    self.info("Evaluate Test over:\n{}\n".format("\n".join(
                        [f"Test {k} = {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in
                         test_metrics.items()])))
                    self.log_metrics({key: val for key, val in dev_metrics.items() if isinstance(val, float)},
                                     prefix='Test')
                    # self.log_metrics({ key : val for key, val in dev_metrics.items() if isinstance(val, float) and not ("wikisql" in key or "annatalk" in key) }, prefix='Test')

                    if not self.args.sampling:
                        self.accelerator.wait_for_everyone()
                        model_saved_path = os.path.join(self.args.output_dir, self.experiment_name,
                                                        "{}.step_{}.pt".format(self.args.model, global_step))
                        self.save(model_saved_path)

        self.info("***** Running training over *****")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', help='learning rate', type=float, default=1e-4)
    parser.add_argument('-plr', '--pretrained_learning_rate', help='pretrained model learning rate', type=float,
                        default=2e-5)
    parser.add_argument('-plm', '--pretrained_model', help='pretrained_model_name', default='tulrv3-small-tab')
    parser.add_argument('-model', '--model', help='Grounding Model Name', default='UniG')
    parser.add_argument('-hs', '--hidden_size', help='hidden size', type=int, default=384)
    parser.add_argument('-bs', '--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('-max_enc_length', '--max_encode_length', help='sequence max encode length', type=int,
                        default=300)
    parser.add_argument('-max_steps', '--max_train_steps', default=10000, type=int)
    parser.add_argument('-glw', '--grounding_loss_weight_func', default='5000_10000', type=str,
                        help='Grounding Loss Weight Function')
    parser.add_argument('-ckpt', '--checkpoint', default=None)
    parser.add_argument('-data_dir', '--data_dir', default=os.getenv("AMLT_DATA_DIR", default='data'))
    parser.add_argument('-datasets', '--datasets', default='wikisql')
    parser.add_argument('-ignore_umv', '--ignore_unmatched_values', default=True, type=bool)
    parser.add_argument('-out_dir', '--output_dir', default=os.getenv("AMLT_OUTPUT_DIR", default='output'))
    parser.add_argument('-eval_steps', '--evaluate_steps', type=int, default=100)
    parser.add_argument('-grad_acc_steps', '--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.3)
    parser.add_argument('-da', '--data_augment', action='store_true')
    parser.add_argument('-ls', '--label_smoothing', type=float, default=0.0)
    parser.add_argument('-use_amp', '--use_amp', action='store_true', default=True)
    parser.add_argument('-use_wb', '--use_wandb', action='store_true')
    parser.add_argument('-exp_id', '--experiment_id', default=None)
    parser.add_argument('-gpu', '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-threads', '--num_threads', default=None, type=int)
    parser.add_argument('-sampling', '--sampling', action='store_true')
    args = parser.parse_args()

    training_args = TrainingArgs(**dict(args._get_kwargs()))
    return training_args


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
