import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam

class WarmupPolynomialLRScheduler:
    optimizer: Optimizer
    num_warmup_steps: int
    start_lr: float
    end_lr: float
    decay_steps: int
    power: float
    
    def __init__(self, optimizer: Optimizer, start_lr: float, num_warmup_steps: int = 2000, end_lr: float = 0.0, decay_steps: int = 98000, power: float = 1.0) -> None:
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.decay_steps = decay_steps
        self.power = power

    def update(self, step: int):
        if step < self.num_warmup_steps:
            warmup_frac_done = step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        elif step < (self.num_warmup_steps + self.decay_steps):
            new_lr = (self.start_lr - self.end_lr) * (
                1 - (step - self.num_warmup_steps) / self.decay_steps
            ) ** self.power + self.end_lr
        else:
            new_lr = self.end_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

class BertWarmupPolynomialLRScheduler(WarmupPolynomialLRScheduler):
    bert_factor: float

    def __init__(self, optimizer: Optimizer, start_lr: float, bert_factor: float, num_warmup_steps: int = 2000, end_lr: float = 0.0, decay_steps: int = 98000, power: float = 1.0) -> None:
        super().__init__(optimizer, start_lr, num_warmup_steps=num_warmup_steps, end_lr=end_lr, decay_steps=decay_steps, power=power)
        self.bert_factor = bert_factor
    
    def update(self, step):
        super(BertWarmupPolynomialLRScheduler, self).update(step)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "bert":
                param_group["lr"] /= self.bert_factor

def _is_bert_parameter(param_name: str):
    return param_name.startswith('bert') or param_name.startswith('encoder.bert')

def get_optimizer_and_lr_scheduler(model: nn.Module, lr: float, num_warmup_steps: int = 2000, bert_factor: float = 8):
    optimizer = Adam(params=[
        {
            "name": "no-bert",
            "params": (
                parameters
                for name, parameters in model.named_parameters()
                if not _is_bert_parameter(name)
            ),
        },
        {
            "name": "bert",
            "params": (
                parameters
                for name, parameters in model.named_parameters()
                if _is_bert_parameter(name)
            ),
        }])
    
    lr_scheduler = BertWarmupPolynomialLRScheduler(optimizer=optimizer, start_lr=lr, bert_factor=bert_factor, num_warmup_steps=num_warmup_steps)
    return optimizer, lr_scheduler