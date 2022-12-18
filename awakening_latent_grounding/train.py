import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from models import *
from utils import *
from datetime import datetime
import logging
from dataclasses import dataclass, field


@dataclass
class TrainingArgs:
    learning_rate: float = field(default=3e-5, metadata={"help": "The initial learning rate for Adam."})
    non_bert_learning_rate: float = field(default=1e-3,
                                          metadata={"help": "The initial learning rate for non-BERT parameters."})

    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    dropout: float = field(default=0.3)
    label_smoothing: bool = field(default=True)
    fp16: bool = field(default=False)

    train_batch_size: int = field(default=16, metadata={"help": "Training batch size"})
    eval_batch_size: int = field(default=32, metadata={"help": "Evaluation batch size"})
    num_train_epochs: int = field(default=10, metadata={"help": "Training epochs"})
    max_encode_length: int = field(default=512)
    warmup_steps: int = field(default=0, metadata={"help": "Warmup steps"})

    alw_func: str = field(default='const_0.0')

    logging_steps: int = field(default=100)
    evaluate_steps: int = field(default=1000)
    accumulation_steps: int = field(default=1)

    model: str = field(default='AlignmentModel')
    bert_version: str = field(default="bert-base-uncased", metadata={"help": "Training epochs"})
    checkpoint: str = field(default=None)

    data_dir: str = field(default="data/slsql", metadata={"help": "input data dir"})
    out_dir: str = field(default='out', metadata={'help': 'output data dir'})
    sampling: bool = field(default=False)
    device: str = field(default='cpu')
    seed: int = field(default=123)


def get_logger(log_dir: str, version: str):
    os.makedirs(log_dir, exist_ok=True)
    logging_file = os.path.join(log_dir, '{}.log'.format(version))
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s\t%(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=logging_file,
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s\t%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logger = logging.getLogger("")
    logger.addHandler(console)
    return logger


def set_seed(seed=123):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    args: TrainingArgs
    model: nn.Module
    logger: logging.Logger
    device: torch.device
    version: str

    def __init__(self, args: TrainingArgs):
        set_seed()

        self.args = args
        self.version = "{}_{}".format(args.model, datetime.now().strftime("%Y%m%d%H%M"))
        if self.args.sampling:
            self.version = "Sampling_" + self.version
            self.args.evaluate_steps = 100

        self.logger = get_logger(os.path.join(self.args.out_dir, self.version), self.args.model)

        self.device = torch.device(args.device)
        self.model = load_model_from_checkpoint(**self.args.__dict__)

        open(os.path.join(self.args.out_dir, self.version, "config.json"), 'w', encoding='utf-8').write(
            json.dumps(self.args.__dict__, indent=4, sort_keys=True) + '\n')
        self.logger.info("save training config over.")
        for key, val in self.args.__dict__.items():
            self.logger.info("{} = {}".format(key, val))

    def get_train_and_eval_iter(self):
        bert_version = self.args.bert_version.replace("hfl/", "")
        train_paths = [os.path.join(self.args.data_dir, f"train.{bert_version}.json")]
        dev_paths = [os.path.join(self.args.data_dir, f"dev.{bert_version}.json")]

        tokenizer = BertTokenizer.from_pretrained(self.args.bert_version)
        self.logger.info("load BERT tokenizer from {} over.".format(self.args.bert_version))

        data_loader_func = get_data_iterator_func(self.args.model)
        if self.args.sampling:
            train_iter = data_loader_func(train_paths, tokenizer, self.args.train_batch_size, self.device, False, True,
                                          self.args.max_encode_length, sampling_size=self.args.train_batch_size * 100)
            dev_iter = data_loader_func(dev_paths, tokenizer, self.args.eval_batch_size, self.device, False, False,
                                        self.args.max_encode_length, sampling_size=self.args.train_batch_size * 20)
        else:
            train_iter = data_loader_func(train_paths, tokenizer, self.args.train_batch_size, self.device, False, True,
                                          self.args.max_encode_length)
            dev_iter = data_loader_func(dev_paths, tokenizer, self.args.eval_batch_size, self.device, False, False,
                                        self.args.max_encode_length)

        self.logger.info("load train iterator over, size = {}".format(len(train_iter.batch_sampler)))
        self.logger.info("load dev iterator over, size = {}".format(len(dev_iter.batch_sampler)))
        return train_iter, dev_iter

    def evaluate(self, dev_iter: DataLoader, saved_file=None):
        model = self.model
        model.eval()
        evaluator = get_evaluator_class(self.args.model)()
        with torch.no_grad():
            for batch_inputs in dev_iter:
                batch_inputs['label_smoothing'] = self.args.label_smoothing
                batch_outputs = model.compute_loss(**batch_inputs)
                evaluator.add_batch(batch_inputs, batch_outputs)

        saved_path = os.path.join(self.args.out_dir, self.version, saved_file) if saved_file is not None else None
        eval_result = evaluator.get_metrics(saved_path)
        model.train()
        self.logger.info("Evaluate over:\n{}".format(
            "\n".join([f"{k} = {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in eval_result.items()])))
        return eval_result

    def _parse_loss_weight_function(self, strategy: str):
        if is_float(strategy):
            w = float(strategy)
            return lambda _: w
        if strategy.startswith("linear"):
            start, end = strategy.replace("linear_", "").split('-')
            start, end = int(start), int(end)
            return lambda x: min(1.0, max(0.0, (x - start) / end))
        else:
            raise NotImplementedError(f"not supported ALW function: {strategy}")

    def train(self):
        train_iter, dev_iter = self.get_train_and_eval_iter()

        num_train_steps = self.args.num_train_epochs * int(len(train_iter))
        self.logger.info("num_train_steps = {}".format(num_train_steps))
        global_step = int(self.args.checkpoint.split('/')[-1].split('.')[1].replace('step_',
                                                                                    "")) if self.args.checkpoint is not None else 0
        start_epoch = global_step // len(train_iter)

        params = []
        params_bert = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bert' in name:
                    params_bert.append(param)
                else:
                    params.append(param)
        optimizer = AdamW([{'params': params_bert},
                           {'params': params, 'lr': 1e-3}],
                          lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            self.logger.info("Enable fp16 optimization ...")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O2")

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_steps)

        self.model.train()

        loss_weight_funcs = {
            'align_loss': self._parse_loss_weight_function(self.args.alw_func),
        }

        grad_accumulation_count = 0
        for epoch in range(start_epoch, self.args.num_train_epochs):
            logging_loss = defaultdict(float)
            for batch_inputs in train_iter:
                global_step += 1
                grad_accumulation_count += 1

                batch_inputs['label_smoothing'] = self.args.label_smoothing
                for loss_type in loss_weight_funcs:
                    batch_inputs[loss_type + "_weight"] = loss_weight_funcs[loss_type](epoch + 1)

                outputs = self.model.compute_loss(**batch_inputs)
                loss = outputs['loss'] / self.args.accumulation_steps
                if self.args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if grad_accumulation_count % self.args.accumulation_steps == 0:
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                logging_loss['total_loss'] += loss.item() * self.args.accumulation_steps
                for loss_type in loss_weight_funcs:
                    if loss_type in outputs:
                        logging_loss[loss_type] += outputs[loss_type].item()

                if global_step % self.args.logging_steps == 0:
                    avg_loss = logging_loss['total_loss'] / self.args.logging_steps
                    loss_string = "total loss: {:.4f}; ".format(avg_loss)
                    for loss_type in loss_weight_funcs:
                        if loss_type in logging_loss:
                            loss_string += "{} : {:.4f} ({:.3f}); ".format(loss_type, logging_loss[
                                loss_type] / self.args.logging_steps, loss_weight_funcs[loss_type](epoch + 1))

                    self.logger.info(
                        "Epoch: {}, Step: {}/{}, {}".format(epoch + 1, global_step, len(train_iter) * (epoch + 1),
                                                            loss_string))
                    logging_loss = defaultdict(float)

                if global_step % self.args.evaluate_steps == 0:
                    self.logger.info("Evaluating step {} ...".format(global_step))
                    eval_metrics = self.evaluate(dev_iter, saved_file='eval.step_{}.txt'.format(global_step))
                    saved_path = os.path.join(self.args.out_dir, self.version, "{}.step_{}.acc_{:.3f}.pt".format(
                        self.args.model,
                        global_step,
                        eval_metrics['overall accuracy']))
                    torch.save(self.model.state_dict(), saved_path)
                    self.logger.info("Save checkpoint to {}".format(saved_path))

        self.logger.info("***** Running training over *****")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', help='learning rate', type=float, default=5e-5)
    parser.add_argument('-model', '--model', help='model', default='SpiderAlignmentModel')
    parser.add_argument('-bert', '--bert_version', help='bert version', default='bert-base-uncased')
    parser.add_argument('-train_bs', '--train_batch_size', help='train batch size', type=int, default=10)
    parser.add_argument('-eval_bs', '--eval_batch_size', help='eval batch size', type=int, default=10)
    parser.add_argument('-max_enc_length', '--max_encode_length', help='sequence max encode length', type=int,
                        default=512)
    parser.add_argument('-num_epochs', '--num_train_epochs', default=30, type=int)
    parser.add_argument('-label_smoothing', '--label_smoothing', action='store_true', default=False)
    parser.add_argument('-sampling', '--sampling', action='store_true')

    # if finetune, use this.
    parser.add_argument('-ckpt', '--checkpoint', default=None)
    parser.add_argument('-alw', '--alw_func', default='0.1')
    parser.add_argument('-data', '--data_dir', default=os.getenv("PT_DATA_DIR", default='data/slsql'))
    parser.add_argument('-out_dir', '--out_dir', default=os.getenv("PT_OUTPUT_DIR", default='pt'))
    parser.add_argument('-acc_steps', '--accumulation_steps', type=int, default=1)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.3)
    parser.add_argument('-fp16', '--fp16', action='store_true')
    parser.add_argument('-gpu', '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    training_args = TrainingArgs(**dict(args._get_kwargs()))
    return training_args


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
    pass
