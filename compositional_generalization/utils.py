import math
import torch
import logging
from functools import partial
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.distributions.utils import lazy_property
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.categorical import Categorical as TorchCategorical
import os
import numpy as np


class VisualizeLogger(object):
    EMOJI_CORRECT = "&#128523;"
    EMOJI_ERROR = "&#128545;"
    EMOJI_REWARD = "🍎"
    EMOJI_DECODE_REWARD = "🍐"

    def __init__(self, summary_dir):
        """

        :param summary_dir: folder to store the tensorboard X log files
        :param validation_size:
        """
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        self.log_writer = SummaryWriter(summary_dir)
        self.global_step = 0
        self.validate_no = 1
        self.validation_size = 1

        # define template
        self.log_template = '**Input**   :   {4} \n\n **Reduce**  :  {1} \n\n **Ground**: {2} \n\n{0}**Logic Form**: {3} \n\n'

    def update_validate_size(self, validation_size):
        self.validation_size = validation_size

    def log_text(self, ground_truth, reduce_str, logic_form, utterance, debug_info=None):
        is_correct = ground_truth == logic_form
        if is_correct:
            logging_str = self.log_template.format(self.EMOJI_CORRECT, reduce_str, ground_truth, logic_form, utterance)
        else:
            logging_str = self.log_template.format(self.EMOJI_ERROR, reduce_str, ground_truth, logic_form, utterance)
        if debug_info is not None:
            tree_vis = self._format_tree_prob(utterance, debug_info)
            logging_str += "**Tree**  :\n\n" + tree_vis
        dev_case = self.global_step % self.validation_size
        dev_step = self.global_step // self.validation_size
        self.log_writer.add_text(f'{dev_case}-th Example', logging_str, global_step=dev_step)

    def log_performance(self, valid_acc):
        self.log_writer.add_scalar("Accuracy", valid_acc, global_step=self.validate_no)

    def update_step(self):
        self.global_step += 1

    def update_epoch(self):
        self.validate_no += 1

    def _format_tree_prob(self, utterance, debug_info):
        # accept utterance and debug_info, return the visualized tree prob
        tokens = utterance.split(" ")
        seq_len = len(tokens)
        merge_probs = debug_info["merge_prob"]
        reduce_probs = debug_info["reduce_prob"]
        decoder_inputs = debug_info["decoder_inputs"]
        decoder_outputs = debug_info["decoder_outputs"]

        reduce_rewards = debug_info["tree_sr_rewards"]
        decode_rewards = debug_info["decode_rewards"]

        log_strs = []
        right_single = "■■"
        error_single = "□□"
        # merged chain
        merge_template = "{3} {0} ({1:.2f}) ({2:.2f})"
        no_merge_template = "{2} {0} ({1:.2f})"
        only_reduce_template = "{0} (1.00) ({1:.2f})"

        start_indicator = 0
        depth_indicator = 0
        decode_indicator = 0
        if seq_len == 1:
            log_str = only_reduce_template.format(tokens[0], reduce_probs[0])
            log_strs.append(log_str)
        else:
            for reverse_len in reversed(range(1, seq_len)):
                if depth_indicator == 0:
                    # reduce single node
                    for i in range(seq_len):
                        log_str = only_reduce_template.format(tokens[i], reduce_probs[i])
                        if decoder_outputs[i] != 'NONE':
                            log_str += " ({1}{0:.2f})".format(reduce_rewards[i], self.EMOJI_REWARD)
                            log_str += " [*input*: {0}, *output*: {1}]".format(decoder_inputs[i], decoder_outputs[i])
                            log_str += " ({1}{0:.2f})".format(decode_rewards[decode_indicator],
                                                              self.EMOJI_DECODE_REWARD)
                            decode_indicator += 1
                        else:
                            log_str += " ({1}{0:.2f})".format(reduce_rewards[i], self.EMOJI_REWARD)
                        log_strs.append(log_str)
                    depth_indicator += 1

                layer_merge_prob = merge_probs[start_indicator: start_indicator + reverse_len]
                start_indicator += reverse_len
                layer_reduce_prob = reduce_probs[seq_len + depth_indicator - 1]
                merge_candidates = ["-".join(tokens[i: i + depth_indicator + 1]) for i in range(reverse_len)]
                ind = np.argmax(layer_merge_prob)
                for i in range(reverse_len):
                    if i == ind:
                        log_str = merge_template.format(merge_candidates[i], layer_merge_prob[i],
                                                        layer_reduce_prob,
                                                        right_single * depth_indicator)
                        if decoder_outputs[seq_len + depth_indicator - 1] != "NONE":
                            log_str += " ({1}{0:.2f})".format(reduce_rewards[seq_len + depth_indicator - 1],
                                                              self.EMOJI_REWARD)
                            log_str += " [*input*: {0}, *output*: {1}]".format(
                                decoder_inputs[seq_len + depth_indicator - 1],
                                decoder_outputs[seq_len + depth_indicator - 1]
                            )
                            log_str += " ({1}{0:.2f})".format(decode_rewards[decode_indicator],
                                                              self.EMOJI_DECODE_REWARD)
                            decode_indicator += 1
                        else:
                            log_str += " ({1}{0:.2f})".format(reduce_rewards[i], self.EMOJI_REWARD)
                    else:
                        log_str = no_merge_template.format(merge_candidates[i], layer_merge_prob[i],
                                                           error_single * depth_indicator)
                    log_strs.append(log_str)
                depth_indicator += 1
        return "\n\n".join(log_strs)


class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(file_name):
    logger = logging.getLogger("general_logger")
    handler = logging.FileHandler(file_name, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%m-%Y %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def get_lr_scheduler(logger, optimizer, mode='max', factor=0.5, patience=10, threshold=1e-4, threshold_mode='rel'):
    def reduce_lr(self, epoch):
        ReduceLROnPlateau._reduce_lr(self, epoch)
        logger.info(f"learning rate is reduced by factor {factor}!")

    lr_scheduler = ReduceLROnPlateau(optimizer, mode, factor, patience, False, threshold, threshold_mode)
    lr_scheduler._reduce_lr = partial(reduce_lr, lr_scheduler)
    return lr_scheduler


def clamp_grad(v, min_val, max_val):
    if v.requires_grad:
        v_tmp = v.expand_as(v)
        v_tmp.register_hook(lambda g: g.clamp(min_val, max_val))
        return v_tmp
    return v


def length_to_mask(length):
    with torch.no_grad():
        batch_size = length.shape[0]
        max_length = length.data.max()
        range = torch.arange(max_length, dtype=torch.int64, device=length.device)
        range_expanded = range[None, :].expand(batch_size, max_length)
        length_expanded = length[:, None].expand_as(range_expanded)
        return (range_expanded < length_expanded).float()


class Categorical:
    def __init__(self, scores, mask=None):
        self.mask = mask
        if mask is None:
            self.cat_distr = TorchCategorical(F.softmax(scores, dim=-1))
            self.n = scores.shape[0]
            self.log_n = math.log(self.n)
        else:
            self.n = self.mask.sum(dim=-1)
            self.log_n = (self.n + 1e-17).log()
            self.cat_distr = TorchCategorical(Categorical.masked_softmax(scores, self.mask))

    @lazy_property
    def probs(self):
        return self.cat_distr.probs

    @lazy_property
    def logits(self):
        return self.cat_distr.logits

    @lazy_property
    def entropy(self):
        if self.mask is None:
            return self.cat_distr.entropy() * (self.n != 1)
        else:
            entropy = - torch.sum(self.cat_distr.logits * self.cat_distr.probs * self.mask, dim=-1)
            does_not_have_one_category = (self.n != 1.0).to(dtype=torch.float32)
            # to make sure that the entropy is precisely zero when there is only one category
            return entropy * does_not_have_one_category

    @lazy_property
    def normalized_entropy(self):
        return self.entropy / (self.log_n + 1e-17)

    def sample(self):
        return self.cat_distr.sample()

    def rsample(self, temperature=None, gumbel_noise=None, eps=1e-5):
        if gumbel_noise is None:
            with torch.no_grad():
                uniforms = torch.empty_like(self.probs).uniform_()
                uniforms = uniforms.clamp(min=eps, max=1 - eps)
                gumbel_noise = -(-uniforms.log()).log()

        elif gumbel_noise.shape != self.probs.shape:
            raise ValueError

        if temperature is None:
            with torch.no_grad():
                scores = (self.logits + gumbel_noise)
                scores = Categorical.masked_softmax(scores, self.mask)
                sample = torch.zeros_like(scores)
                sample.scatter_(-1, scores.argmax(dim=-1, keepdim=True), 1.0)
                return sample, gumbel_noise
        else:
            scores = (self.logits + gumbel_noise) / temperature
            sample = Categorical.masked_softmax(scores, self.mask)
            return sample, gumbel_noise

    def log_prob(self, value):
        if value.dtype == torch.long:
            if self.mask is None:
                return self.cat_distr.log_prob(value)
            else:
                return self.cat_distr.log_prob(value) * (self.n != 0.).to(dtype=torch.float32)
        else:
            max_values, mv_idxs = value.max(dim=-1)
            relaxed = (max_values - torch.ones_like(max_values)).sum().item() != 0.0
            if relaxed:
                raise ValueError("The log_prob can't be calculated for the relaxed sample!")
            return self.cat_distr.log_prob(mv_idxs) * (self.n != 0.).to(dtype=torch.float32)

    @staticmethod
    def masked_softmax(logits, mask):
        """
        This method will return valid probability distribution for the particular instance if its corresponding row
        in the `mask` matrix is not a zero vector. Otherwise, a uniform distribution will be returned.
        This is just a technical workaround that allows `Categorical` class usage.
        If probs doesn't sum to one there will be an exception during sampling.
        """
        if mask is not None:
            probs = F.softmax(logits, dim=-1) * mask
            probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.).to(dtype=torch.float32)
            Z = probs.sum(dim=-1, keepdim=True)
            return probs / Z
        else:
            return F.softmax(logits, dim=-1)
