import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from models import *
from utils import *
from datetime import datetime
import logging
from dataclasses import dataclass, field

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

def set_seed(seed = 123):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class DistillingArgs:
    learning_rate: float = field(default=3e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    dropout: float = field(default=0.3)

    train_batch_size: int = field(default=16, metadata={"help": "Training batch size"})
    eval_batch_size: int = field(default=32, metadata={"help": "Evaluation batch size"})
    num_train_epochs: int = field(default=10, metadata={"help": "Training epochs"})
    max_encode_length: int = field(default=512)

    alw_func: str = field(default='0.0')
    
    logging_steps : int = field(default=100)
    evaluate_steps: int = field(default=1000)
    accumulation_steps: int = field(default=1)

    checkpoint: str = field(default=None) # pretrained checkpoint
    with_alignments: bool = field(default=False)

    bert_version: str = field(default=None)
    model: str = field(default=None)

    data_dir: str = field(default="data/slsql", metadata={"help": "input data dir"})
    out_dir: str = field(default='out', metadata={'help': 'output data dir'})
    sampling: bool = field(default=False)

    device: str = field(default='cpu')
    seed: int = field(default=123)


class SelfLearningDistiller:
    """
    Teacher-Student Self-Learning Distiller
    """
    args: DistillingArgs
    model: nn.Module
    logger: logging.Logger
    device: torch.device
    version: str

    def __init__(self, args: DistillingArgs) -> None:
        set_seed(args.seed)
        
        self.args = args
        self.version = "self-learning_{}".format(datetime.now().strftime("%Y%m%d%H%M"))
        if self.args.sampling:
            self.version = "sampling_" + self.version
            self.args.evaluate_steps = 100
        
        self.device = torch.device(self.args.device)
        self.logger = get_logger(os.path.join(self.args.out_dir, self.version), 'self_learning')

        self.model = self.load_model_from_ckpt()

        open(os.path.join(self.args.out_dir, self.version, "config.json"), 'w', encoding='utf-8').write(json.dumps(self.args.__dict__, indent=4, sort_keys=True) + '\n')
        self.logger.info("save training config over.")
        
    def load_model_from_ckpt(self):
        ckpt_path = os.path.join(self.args.data_dir, 'checkpoints', self.args.checkpoint)
        ckpt_dir = os.path.dirname(ckpt_path)
        config = json.load(open(os.path.join(ckpt_dir, 'config.json'), 'r', encoding='utf-8'))
        config['dropout'] = self.args.dropout

        assert self.args.bert_version is None or self.args.bert_version == config['bert_version']
        assert self.args.model is None or self.args.model == config['model']
        self.args.bert_version = config['bert_version']
        self.args.model = config['model']

        model = load_model_from_checkpoint(config['model'], self.device, checkpoint=ckpt_path, **{'bert_version': config['bert_version'], 'dropout': 0.0 })
        self.logger.info("Load checkpoint from {} over.".format(self.args.checkpoint, self.args.bert_version))
        
        for key, val in config.items():
            self.logger.info("{} = {}".format(key, val))

        return model
    
    def get_train_and_eval_iter(self):
        bert_version = self.args.bert_version.replace("hfl/", "")
        train_paths = [os.path.join(self.args.data_dir, f"train.{bert_version}.json")]
        dev_paths = [os.path.join(self.args.data_dir, f"dev.{bert_version}.json")]

        tokenizer = BertTokenizer.from_pretrained(self.args.bert_version)
        self.logger.info("load BERT tokenizer from {} over.".format(self.args.bert_version))

        data_loader_func = get_data_iterator_func(self.args.model)
        if self.args.sampling:
            train_iter = data_loader_func(train_paths, tokenizer, self.args.train_batch_size, self.device, False, True, self.args.max_encode_length, sampling_size=self.args.train_batch_size * 100)
            dev_iter = data_loader_func(dev_paths, tokenizer, self.args.eval_batch_size, self.device, False, False, self.args.max_encode_length, sampling_size=self.args.train_batch_size * 20)
        else:
            train_iter = data_loader_func(train_paths, tokenizer, self.args.train_batch_size, self.device, False, True, self.args.max_encode_length)
            dev_iter = data_loader_func(dev_paths, tokenizer, self.args.eval_batch_size, self.device, False, False, self.args.max_encode_length)
        
        self.logger.info("load train iterator over, size = {}".format(len(train_iter.batch_sampler)))
        self.logger.info("load dev iterator over, size = {}".format(len(dev_iter.batch_sampler)))
        return train_iter, dev_iter

    def evaluate(self, model: nn.Module, dev_iter: DataLoader, saved_file=None):
        model.eval()
        evaluator = get_evaluator_class(self.args.model)()
        with torch.no_grad():
            for batch_inputs in dev_iter:
                batch_outputs = model.compute_loss(**batch_inputs)
                evaluator.add_batch(batch_inputs, batch_outputs)

        saved_path = os.path.join(self.args.out_dir, self.version, saved_file) if saved_file is not None else None
        eval_result = evaluator.get_metrics(saved_path)
        model.train()
        self.logger.info("Evaluate over:\n{}".format("\n".join([f"{k} = {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in eval_result.items()])))
        return eval_result

    @staticmethod
    def get_masking_inference_func(masking_inputs: Dict, model: nn.Module, infer_size: int):
        infer_outputs = defaultdict(list)
        input_token_ids, input_token_types, meta_index = masking_inputs['input_token_ids'], masking_inputs['input_token_types'], masking_inputs['meta_index']

        model.eval()
        index = 0
        with torch.no_grad():
            while index < len(input_token_ids):
                model_inputs = {
                    'input_token_ids': input_token_ids[index:index+infer_size],
                    'input_token_types': input_token_types[index:index+infer_size],
                    'meta_index': meta_index[index:index+infer_size]
                }

                model_outputs = model.forward(**model_inputs)
                for token_type in [SQLTokenType.table, SQLTokenType.column, SQLTokenType.value]:
                    if f'{str(token_type)}_logits' in model_outputs:
                        infer_outputs[token_type.abbr] += model_outputs[f'{str(token_type)}_logits']

                index += infer_size
        
        for key, val in infer_outputs.items():
            infer_outputs[key] = torch.stack(val, dim=0)
        
        return infer_outputs
    
    @staticmethod
    def get_alignment_weights_from_teacher(inputs: Dict, teacher: nn.Module):
        teacher.eval()
        with torch.no_grad():
            outputs = teacher.forward(**inputs)
            assert 'alignment_weights' in outputs
            alignment_weights: List[torch.Tensor] = outputs['alignment_weights']
            for i in range(len(alignment_weights)):

                labels = torch.cat([inputs['table_labels'][i], inputs['column_labels'][i], inputs['value_labels'][i]], dim=0)
                assert alignment_weights[i].size(0) == len(labels)
                alignment_weights[i].masked_fill_((labels == 0)[:, None], 0.0)

            return alignment_weights

    @staticmethod
    def soft_cross_entropy_with_logits(predict_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        return (- target_logits * predict_logits.log()).mean()

    def _compute_distill_loss(self, s_outputs, t_alignments: torch.Tensor):
        align_ce_loss = 0
        for i in range(len(t_alignments)):
            align_ce_loss += self.soft_cross_entropy_with_logits(
                s_outputs['alignment_weights'][i],
                t_alignments[i]
            )
        
        align_ce_loss /= len(t_alignments)
        s_outputs['identify_loss'] = s_outputs['loss']
        s_outputs['align_loss'] = align_ce_loss
        s_outputs['loss'] = s_outputs['identify_loss'] + s_outputs['align_loss'] * float(self.args.alw_func)
        return s_outputs

    def distill(self):
        """
        use pre-trained student model as teacher to train a new student model
        """
        train_iter, dev_iter = self.get_train_and_eval_iter()

        teacher = self.model
        self.logger.info("Evaluating Teacher ...")
        self.evaluate(teacher, dev_iter, 'teacher.eval.txt')
        teacher.eval()

        model_args = { 'bert_version': self.args.bert_version, 'dropout': self.args.dropout }
        student = load_model_from_checkpoint(model=self.args.model, device=self.args.device, **model_args)
        self.logger.info("Initialize new model as student over.")

        num_train_steps = self.args.num_train_epochs * int(len(train_iter))
        self.logger.info("num_train_steps = {}".format(num_train_steps))
        
        optimizer = AdamW(student.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * num_train_steps), num_training_steps=num_train_steps)
        
        student.train()

        grad_accumulation_count = 0
        global_step = 0
        best_ckpt_path, best_eval_score = None, -100

        for epoch in range(self.args.num_train_epochs):
            logging_loss = defaultdict(float)
            for batch_inputs in train_iter:
                global_step += 1
                grad_accumulation_count += 1

                if not self.args.with_alignments:
                    batch_inputs['align_loss_weight'] = float(self.args.alw_func)
                    batch_inputs['masking_infer_func'] = lambda x: self.get_masking_inference_func(x, teacher, self.args.train_batch_size)
                    outputs = student.compute_loss(**batch_inputs)
                else:
                    outputs = student.compute_loss(**batch_inputs)
                    teacher_alignment_weights = self.get_alignment_weights_from_teacher(batch_inputs, teacher)
                    outputs = self._compute_distill_loss(outputs, teacher_alignment_weights)

                loss = outputs['loss'] / self.args.accumulation_steps
                loss.backward()

                if grad_accumulation_count % self.args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                logging_loss['total_loss'] += loss.item() * self.args.accumulation_steps
                logging_loss['align_loss'] += outputs['align_loss'].item()

                if global_step % self.args.logging_steps == 0:
                    loss_string = "total loss: {:.4f}; align loss: {:.4f} ({:.4f})".format(
                        logging_loss['total_loss'] / self.args.logging_steps,
                        logging_loss['align_loss'] / self.args.logging_steps,
                        float(self.args.alw_func))        

                    self.logger.info("Epoch: {}, Step: {}/{}, {}".format(epoch + 1, global_step, len(train_iter) * (epoch + 1), loss_string))
                    logging_loss = defaultdict(float)

                if global_step % self.args.evaluate_steps == 0:
                    self.logger.info("Evaluating student step {} ...".format(global_step))
                    eval_metrics = self.evaluate(student, dev_iter, saved_file='student.eval.step_{}.txt'.format(global_step))
                    eval_score = (eval_metrics['overall accuracy'] + eval_metrics['average F1']) / 2

                    saved_path = os.path.join(self.args.out_dir, self.version, "student.step_{}.acc_{:.3f}.f1_{:.3f}.pt".format(
                        global_step,
                        eval_metrics['overall accuracy'],
                        eval_metrics['average F1']))

                    torch.save(student.state_dict(), saved_path)
                    self.logger.info("Save checkpoint to {}".format(saved_path))

                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        best_ckpt_path = saved_path

        self.logger.info("Best Student Model Path: {}".format(best_ckpt_path))
        self.logger.info("***** Running teacher-student self-training over *****")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', help='learning rate', type=float, default=5e-5)
    parser.add_argument('-train_bs', '--train_batch_size', help='train batch size', type=int, default=10)
    parser.add_argument('-eval_bs', '--eval_batch_size', help='eval batch size', type=int, default=10)
    parser.add_argument('-max_enc_length', '--max_encode_length', help='sequence max encode length', type=int, default=512)
    parser.add_argument('-num_epochs', '--num_train_epochs', default=30, type=int)
    parser.add_argument('-sampling', '--sampling', action='store_true')

    parser.add_argument('-ckpt', '--checkpoint', default=None)
    parser.add_argument('-with_align', '--with_alignments', action='store_true')
    parser.add_argument('-alw', '--alw_func', default='0.1')

    parser.add_argument('-data', '--data_dir', default=os.getenv("PT_DATA_DIR", default='data/slsql'))
    parser.add_argument('-out_dir', '--out_dir', default=os.getenv("PT_OUTPUT_DIR",default='pt'))
    parser.add_argument('-acc_steps', '--accumulation_steps', type=int, default=1)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.3)
    parser.add_argument('-gpu', '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    distill_args = DistillingArgs(**dict(args._get_kwargs()))
    return distill_args

if __name__  == '__main__':
    args = parse_args()
    distiller = SelfLearningDistiller(args)
    distiller.distill()
