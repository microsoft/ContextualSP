# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import copy
import imp
import sys, os
import torch
import tasks
import math
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from data_utils.utils import AverageMeter
from mt_dnn.loss import LOSS_REGISTRY
from mt_dnn.matcher import SANBertNetwork
from mt_dnn.batcher import Collater
from mt_dnn.perturbation import SmartPerturbation
from mt_dnn.loss import *
from mt_dnn.optim import AdamaxW
from data_utils.task_def import TaskType, EncoderModelType
from experiments.exp_def import TaskDef
from data_utils.my_statics import DUMPY_STRING_FOR_EMPTY_ANS
from transformers.modeling_utils import unwrap_model
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_parameter_names


logger = logging.getLogger(__name__)


def calculate_cosine_similarity(task_grads):
    dot_prod = torch.mm(task_grads, task_grads.t())
    norm = torch.norm(task_grads, p=2, dim=1).unsqueeze(0)
    cos = dot_prod.div(torch.mm(norm.t(), norm))
    return cos


class MTDNNModel(object):
    def __init__(self, 
                 opt, 
                device=None, 
                state_dict=None, 
                num_train_step=-1, 
                adapter=False, 
                adapter_args=None, 
                task_name='adapter', 
                id_task_map=None, 
                heldout_eval_dataset=None):
        self.config = opt
        self.updates = (
            state_dict["updates"] if state_dict and "updates" in state_dict else 0
        )
        self.id_task_map = id_task_map
        self.heldout_eval_dataset = heldout_eval_dataset
        self.adapter = adapter
        self.adapter_cache_path = adapter_args.adapter_cache_path
        self.min_intra_simiarity = adapter_args.min_intra_simiarity
        self.max_entropy_threshold = adapter_args.max_entropy_threshold
        self.max_interference_degree = adapter_args.max_interference_degree
        self.train_adapter_fusion = False
        self.entropy_validate = False
        self.local_updates = 0
        self.device = device
        self.train_loss = AverageMeter()
        self.adv_loss = AverageMeter()
        self.emb_val = AverageMeter()
        self.eff_perturb = AverageMeter()
        self.initial_from_local = True if state_dict else False
        model = SANBertNetwork(opt, initial_from_local=self.initial_from_local, adapter_args=adapter_args, adapter=adapter, task_name=task_name)

        self.diff_task_names = task_name.split('-')
        self.current_task = self.diff_task_names[0]
        if adapter_args.adapter_diff:
            self.laryerwise_candidate_adapter = dict()
            self.current_active_adapters = dict()
            for i in range(len(model.bert.encoder.layer)):
                self.laryerwise_candidate_adapter[f'L{str(i)}'] = \
                    dict([(name, f'{task_name}-L{str(i)}') for name in self.diff_task_names])
                self.current_active_adapters[f'L{str(i)}'] = f'{task_name}-L{str(i)}'
        
        # train adapter fusion with adapter differentiation
        if self.train_adapter_fusion:
            fusion_active = False
            self.laryerwise_fusion_adapters = dict()
            for i in range(model.bert.encoder.layer):
                self.laryerwise_fusion_adapters[f'L{str(i)}'] = [fusion_active, f'Fusion-L{str(i)}']

        self.total_param = sum(
            [p.nelement() for p in model.parameters() if p.requires_grad]
        )
        if opt["cuda"]:
            if self.config["local_rank"] != -1:
                model = model.to(self.device)
            else:
                model = model.to(self.device)
        self.network = model
        if state_dict:
            missing_keys, unexpected_keys = self.network.load_state_dict(
                state_dict["state"], strict=False
            )

        optimizer_parameters = self._get_param_groups()

        self._setup_optim(optimizer_parameters, state_dict, num_train_step)
        self.optimizer.zero_grad()

        # if self.config["local_rank"] not in [-1, 0]:
        #    torch.distributed.barrier()

        if self.config["local_rank"] != -1:
            self.mnetwork = torch.nn.parallel.DistributedDataParallel(
                self.network,
                device_ids=[self.config["local_rank"]],
                output_device=self.config["local_rank"],
                find_unused_parameters=True,
            )
        elif self.config["multi_gpu_on"]:
            self.mnetwork = nn.DataParallel(self.network)
        else:
            self.mnetwork = self.network
        self._setup_lossmap(self.config)
        self._setup_kd_lossmap(self.config)
        self._setup_adv_lossmap(self.config)
        self._setup_adv_training(self.config)
        self._setup_tokenizer()

    # Adapter Differentiation
    def _switch_model_task_mode(self, target_task):
        # Switch the model on the target task mode
        self.current_task = target_task

        if not self.train_adapter_fusion:
            adapter_names = []
            for layer, adapter_name in self.current_active_adapters.items():
                if adapter_name.startswith(f'{target_task}-') or f'-{target_task}-' in adapter_name:
                    adapter_names.append(adapter_name)
                    continue
                else:
                    if len(adapter_name) > 0:
                        self._deactivate_adapter_runtime(adapter_name)
                    target_adapter = self.laryerwise_candidate_adapter[layer][target_task]
                    adapter_names.append(target_adapter)
                    self._activate_adapter_runtime(target_adapter)
                    self.current_active_adapters[layer] = target_adapter
            
            # print('Switch to', target_task, adapter_names)
            self.mnetwork.bert.train_adapter(adapter_names)
        else:
            adapter_fusion_names = []
            undiff_adapters = []
            for layer, fusion_state in self.laryerwise_fusion_adapters.items():
                if fusion_state[0]:
                    adapter_fusion_names.append(fusion_state[1].split(','))
                else:
                    undiff_adapter = self.laryerwise_candidate_adapter[layer][self.current_task]
                    undiff_adapters.append(undiff_adapter)

            if len(adapter_fusion_names) > 0:
                self.mnetwork.bert.train_adapter_and_fusion(undiff_adapters, adapter_fusion_names, unfreeze_adapters=True, target_task=target_task)
            else:
                self.mnetwork.bert.train_adapter(undiff_adapters)

        if torch.cuda.is_available():
            self.mnetwork.bert = self.mnetwork.bert.to(self.device)

        self.update_optimizer_params_groups()

    def _extract_adapter_grads(self, heldout_eval_datasets):
        # record all the adapter gradients on held-out evaluation data
        heldout_evl_nums = len(heldout_eval_datasets)
        exist_adapter_cell_grads = [dict() for _ in range(heldout_evl_nums)]

        # TODO
        heldout_dataloaders = heldout_eval_datasets

        for current_task in self.diff_task_names:
            self._switch_model_task_mode(current_task)
            for hi, heldout_dataloader in enumerate(heldout_dataloaders):
                self.optimizer.zero_grad()

                for _, (batch_meta, batch_data) in enumerate(heldout_dataloader):
                    batch_meta, batch_data = Collater.patch_data(self.device, batch_meta, batch_data)
                    task_name = self.id_task_map[batch_meta["task_id"]]
                    if task_name != current_task:
                        continue
                    
                    # Calculate the gradient of the given heldout evaluation data
                    loss = self.compute_loss(batch_meta, batch_data)
                    loss.backward()

                    if self.entropy_validate:
                        tmp_adapter_cell_grads = dict()
                        for name, param in self.mnetwork.bert.named_parameters():
                            if 'adapter' in name and (f'.{current_task}-' in name or f'-{current_task}-' in name) and ',' not in name and param.requires_grad:
                                layer = name.split('.')[5].split('-')[-1]
                                if layer not in tmp_adapter_cell_grads:
                                    tmp_adapter_cell_grads[layer] = {}
                                if current_task not in tmp_adapter_cell_grads[layer]:
                                    tmp_adapter_cell_grads[layer][current_task] = []

                                tmp_adapter_cell_grads[layer][current_task].append(param.grad.clone().detach().view(1, -1))
                        
                        for layer, task_grads in tmp_adapter_cell_grads.items():
                            for task, task_grad in task_grads.items():
                                cat_grad = torch.cat(task_grad, dim=1)

                                if layer not in exist_adapter_cell_grads[hi]:
                                    exist_adapter_cell_grads[hi][layer] = {}
                                if task not in exist_adapter_cell_grads[hi][layer]:
                                    exist_adapter_cell_grads[hi][layer][task] = []

                                exist_adapter_cell_grads[hi][layer][task].append(cat_grad)
                        
                        self.optimizer.zero_grad()

                if not self.entropy_validate:
                    for name, param in self.mnetwork.bert.named_parameters():
                        if 'adapter' in name and (f'.{current_task}-' in name or f'-{current_task}-' in name) and ',' not in name and param.requires_grad:
                            layer = name.split('.')[5].split('-')[-1]
                            if layer not in exist_adapter_cell_grads[hi]:
                                exist_adapter_cell_grads[hi][layer] = {}
                            if current_task not in exist_adapter_cell_grads[hi][layer]:
                                exist_adapter_cell_grads[hi][layer][current_task] = []

                            exist_adapter_cell_grads[hi][layer][current_task].append(param.grad.clone().detach().view(1, -1))
        
        return exist_adapter_cell_grads
    
    def _find_differentiatable_cell(self, exist_adapter_cell_grads):
        # find all the differentiatable cells according to the
        # MAIN algorithem in the paper
        # output: {'original_cell': List(differentiated cells)}
        # update global active cells
        def _calculate_interference_degree(task_grad_mapping):
            shared_task_len = len(task_grad_mapping)

            assert shared_task_len > 1

            task_grads = torch.stack([g.view(-1,) for g in task_grad_mapping.values()])
            cos = calculate_cosine_similarity(task_grads)

            interference_degree = []
            for i in range(shared_task_len-1):
                # interference degree equals to nagetive cosine similarity
                interference_degree.append(-cos[i, i+1:])
            interference_degree = torch.cat(interference_degree)

            return list(task_grad_mapping.keys()), interference_degree

        # To alleviate over-differentiaiton problem
        if not self.entropy_validate:
            laryerwise_adapter_grad_mappings = []
            for exist_adapter_cell_grad in exist_adapter_cell_grads:
                laryerwise_adapter_grad_mapping = {}
                for layer, task_grads in exist_adapter_cell_grad.items():
                    for task_name, task_grad in task_grads.items():
                        adapter_name = self.laryerwise_candidate_adapter[layer][task_name]
                        if adapter_name not in laryerwise_adapter_grad_mapping:
                            laryerwise_adapter_grad_mapping[adapter_name] = {}

                        task_grad = torch.cat(task_grad, dim=1)
                        laryerwise_adapter_grad_mapping[adapter_name][task_name] = task_grad
                laryerwise_adapter_grad_mappings.append(laryerwise_adapter_grad_mapping)

            if len(laryerwise_adapter_grad_mappings) == 1:
                merge_laryerwise_adapter_grad_mapping = laryerwise_adapter_grad_mappings[0]
            else:
                assert len(laryerwise_adapter_grad_mappings) == 2
                merge_laryerwise_adapter_grad_mapping = dict()
                laryerwise_adapter_grad_mapping1 = laryerwise_adapter_grad_mappings[0]
                laryerwise_adapter_grad_mapping2 = laryerwise_adapter_grad_mappings[1]

                differentiable_adapters = []
                for adapter_name, task_grad_mapping in laryerwise_adapter_grad_mapping1.items():
                    if len(task_grad_mapping) > 1:
                        diff_flag = True
                        for task, grad in task_grad_mapping.items():
                            aux_grad = laryerwise_adapter_grad_mapping2[adapter_name][task]

                            grad_ = grad.view(1,-1)
                            aux_grad_ = aux_grad.view(1,-1)
                            dot_prod = calculate_cosine_similarity(torch.stack([grad_.view(-1,), aux_grad_.view(-1,)]))
                            dot_prod = dot_prod[0][1]
                            print('>>>> dot_prod', dot_prod)
                            if dot_prod < math.cos(math.pi / self.min_intra_simiarity):
                                diff_flag = False
                                break
                        
                        if diff_flag:
                            differentiable_adapters.append(adapter_name)

                for adapter_name in differentiable_adapters:
                    merge_laryerwise_adapter_grad_mapping[adapter_name] = dict()
                    
                    for task, grad in laryerwise_adapter_grad_mapping1[adapter_name].items():
                        aux_grad = laryerwise_adapter_grad_mapping2[adapter_name][task]
                        merge_laryerwise_adapter_grad_mapping[adapter_name][task] = grad + aux_grad
        else:
            merge_laryerwise_adapter_grad_mapping = dict()
            laryerwise_adapter_grad_mappings = []
            for exist_adapter_cell_grad in exist_adapter_cell_grads:
                laryerwise_adapter_grad_mapping = {}
                for layer, task_grads in exist_adapter_cell_grad.items():
                    for task_name, task_grad in task_grads.items():
                        adapter_name = self.laryerwise_candidate_adapter[layer][task_name]
                        if adapter_name not in laryerwise_adapter_grad_mapping:
                            laryerwise_adapter_grad_mapping[adapter_name] = {}

                        task_grad = torch.cat(task_grad, dim=0)
                        laryerwise_adapter_grad_mapping[adapter_name][task_name] = task_grad
                laryerwise_adapter_grad_mappings.append(laryerwise_adapter_grad_mapping)

            if len(laryerwise_adapter_grad_mappings) == 1:
                laryerwise_adapter_grad_mapping = laryerwise_adapter_grad_mappings[0]
            else:
                assert len(laryerwise_adapter_grad_mappings) == 2
                laryerwise_adapter_grad_mapping1 = laryerwise_adapter_grad_mappings[0]
                laryerwise_adapter_grad_mapping2 = laryerwise_adapter_grad_mappings[1]

                for adapter_name, task_grad_mapping in laryerwise_adapter_grad_mapping1.items():
                    if len(task_grad_mapping) > 1:
                        for task, grad in task_grad_mapping.items():
                            aux_grad = laryerwise_adapter_grad_mapping2[adapter_name][task]
                            laryerwise_adapter_grad_mapping1[adapter_name][task] = torch.cat([grad, aux_grad], dim=0)
                laryerwise_adapter_grad_mapping = laryerwise_adapter_grad_mapping1

            differentiable_adapters = []
            for adapter_name, task_grads in laryerwise_adapter_grad_mapping.items():

                diff_flag = True
                for task, grad in task_grads.items():
                    ave_grad = torch.mean(grad, dim=0)
                    hd_size = grad.size(0)

                    positive_grad = 0
                    for hd_i in range(hd_size):
                        grad_ = grad[hd_i]
                        cos = calculate_cosine_similarity(torch.stack([grad_.view(-1,), ave_grad.view(-1,)]))
                        if cos[0][1] > math.cos(math.pi / self.max_entropy_threshold):
                            positive_grad += 1

                    positive_prob = positive_grad / hd_size

                    if positive_prob == 0 or positive_prob == 1:
                        task_entropy = 0
                    else:
                        task_entropy = -(positive_prob * math.log(positive_prob, 2) + (1-positive_prob) * math.log(1-positive_prob, 2))
                    
                    if task_entropy > -(0.8 * math.log(0.8, 2) + 0.2 * math.log(0.2, 2)) or positive_prob < 0.5:
                        diff_flag = False
                        break
            
                if diff_flag:
                    differentiable_adapters.append(adapter_name)

            for adapter_name in differentiable_adapters:
                merge_laryerwise_adapter_grad_mapping[adapter_name] = dict()
                
                for task, grad in laryerwise_adapter_grad_mapping[adapter_name].items():
                    merge_laryerwise_adapter_grad_mapping[adapter_name][task] = torch.mean(grad, dim=0)

        differentiated_cell_mapping = {}
        for adapter_name, task_grad_mapping in merge_laryerwise_adapter_grad_mapping.items():
            if len(task_grad_mapping) == 1:
                continue
            else:
                tasks, interference_degrees = _calculate_interference_degree(task_grad_mapping)
                max_interference_degree = torch.max(interference_degrees)
                print('>>> max_interference_degree', max_interference_degree)
                task_len = len(tasks)
                if max_interference_degree > self.max_interference_degree:
                    if layer not in differentiated_cell_mapping:
                        differentiated_cell_mapping[adapter_name] = {}
                    # start to differentiate
                    flag = 0
                    group1 = []
                    group2 = []
                    task_distance = {}
                    for i in range(task_len-1):
                        for j in range(i+1, task_len):
                            if interference_degrees[flag] == max_interference_degree:
                                group1.append(i)
                                group2.append(j)
                            
                            task_distance[(i,j)] = interference_degrees[flag]
                            flag += 1

                    for i in range(task_len):
                        if i in group1 or i in group2:
                            continue
                        distance_to_g1 = []
                        for j in group1:
                            a = i
                            b = j
                            if i == j:
                                continue
                            if i > j:
                                a,b = b,a
                            distance_to_g1.append(task_distance[(a,b)])

                        distance_to_g2 = []
                        for k in group2:
                            a = i
                            b = k
                            if i == k:
                                continue
                            if i > k:
                                a,b = b,a
                            distance_to_g2.append(task_distance[(a,b)])
                        
                        distance_to_g1 = torch.stack(distance_to_g1).view(-1,)
                        distance_to_g2 = torch.stack(distance_to_g2).view(-1,)

                        if torch.max(distance_to_g1) < torch.max(distance_to_g2):
                            group1.append(i)
                        else:
                            group2.append(i)

                    group1 = [tasks[t] for t in group1]
                    group2 = [tasks[t] for t in group2]

                    differentiated_cell_mapping[adapter_name] = [group1, group2]
        
        # print('>>> differentiated_cell_mapping', differentiated_cell_mapping)
        return differentiated_cell_mapping

    def _update_differentiated_model(self, differentiated_cells):
        # add new differentiated cells in the model and load 
        # the corresponding params in the optimizer
        for adapter_name, split_group in differentiated_cells.items():
            layer = adapter_name.split('-')[-1]
            adapter_group1 = '-'.join(split_group[0]) + f'-{layer}'
            adapter_group2 = '-'.join(split_group[1]) + f'-{layer}'

            if adapter_name == self.current_active_adapters[layer]:
                self._deactivate_adapter_runtime(adapter_name)
            self._copy_adapter_runtime(adapter_group1, adapter_name)
            self._copy_adapter_runtime(adapter_group2, adapter_name)

            self.current_active_adapters[layer] = ''

            for task in split_group[0]:
                self.laryerwise_candidate_adapter[layer][task] = adapter_group1

            for task in split_group[1]:
                self.laryerwise_candidate_adapter[layer][task] = adapter_group2

    def _update_differentiated_fusion_model(self, differentiated_cells):
        # add new differentiated cells in the model and load 
        # the corresponding params in the optimizer

        processed_fusion_layer = []
        for adapter_name, split_group in differentiated_cells.items():
            layer = adapter_name.split('-')[-1]
            adapter_group1 = '-'.join(split_group[0]) + f'-{layer}'
            adapter_group2 = '-'.join(split_group[1]) + f'-{layer}'

            layer_fusion_active = self.laryerwise_fusion_adapters[layer][0]

            self._deactivate_adapter_runtime(adapter_name)
            if layer_fusion_active and layer not in processed_fusion_layer:
                self._deactivate_adapter_fusion_runtime(self.laryerwise_fusion_adapters[layer][1])

            if layer not in processed_fusion_layer:
                processed_fusion_layer.append(layer)
            
            self._copy_adapter_runtime(adapter_group1, adapter_name)
            self._copy_adapter_runtime(adapter_group2, adapter_name)

            for task in split_group[0]:
                self.laryerwise_candidate_adapter[layer][task] = adapter_group1

            for task in split_group[1]:
                self.laryerwise_candidate_adapter[layer][task] = adapter_group2
        
        for layer in processed_fusion_layer:
            layer_fusion_active = self.laryerwise_fusion_adapters[layer][0]

            layer_adapters = list(set(list(self.laryerwise_candidate_adapter[layer].values())))
            layer_fusion_name = ','.join(layer_adapters)
            if not layer_fusion_active:
                self._create_adapter_fusion_runtime(layer_fusion_name)
                self.laryerwise_fusion_adapters[layer][0] = True
            else:
                self._copy_adapter_fusion_runtime(layer_fusion_name, self.laryerwise_fusion_adapters[layer][1])
            
            self.laryerwise_fusion_adapters[layer][1] = layer_fusion_name

    def _differentiate_operate(self):
        exist_adapter_cell_grads = self._extract_adapter_grads(self.heldout_eval_dataset)
        diff_cells = self._find_differentiatable_cell(exist_adapter_cell_grads)
        print(diff_cells)

        if not self.train_adapter_fusion:
            self._update_differentiated_model(diff_cells)
        else:
            self._update_differentiated_fusion_model(diff_cells)

    def _calculate_differentiated_rate(self):
        initial_adapter_num = len(self.laryerwise_candidate_adapter)
        current_adapter_names = []
        for layer in self.laryerwise_candidate_adapter.keys():
            for task_name in self.laryerwise_candidate_adapter[layer].keys():
                current_adapter_names.append(self.laryerwise_candidate_adapter[layer][task_name])
        
        current_adapter_num = len(list(set(current_adapter_names)))

        return current_adapter_num / initial_adapter_num

    def update_optimizer_params_groups(self):
        decay_parameters = get_parameter_names(self.mnetwork.bert, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        add_decay_params = [p for n, p in self.mnetwork.bert.named_parameters() if n in decay_parameters and p.requires_grad and n not in self.candidate_params]
        add_not_decay_params = [p for n, p in self.mnetwork.bert.named_parameters() if n not in decay_parameters and p.requires_grad and n not in self.candidate_params]

        for n, p in self.mnetwork.bert.named_parameters():
            if p.requires_grad and n not in self.candidate_params:
                self.candidate_params.append(n)

        optimizer_grouped_parameters = []
        if len(add_decay_params) > 0:
            optimizer_grouped_parameters.append(
                {
                    "params": add_decay_params,
                    "weight_decay": self.config['weight_decay'],
                }
            )
        
        if len(add_not_decay_params) > 0:
            optimizer_grouped_parameters.append(
                {
                    "params": add_not_decay_params,
                    "weight_decay": 0.0,
                }
            )

        for param_group in optimizer_grouped_parameters:
            self.optimizer.add_param_group(param_group)

    def _deactivate_adapter_runtime(self, adapter_name):
        save_path = os.path.join(self.adapter_cache_path, adapter_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.mnetwork.bert.save_adapter(save_path, adapter_name)
        self.mnetwork.bert.delete_adapter(adapter_name)

    def _deactivate_adapter_fusion_runtime(self, adapter_fusion_name):
        save_path = os.path.join(self.adapter_cache_path, adapter_fusion_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.mnetwork.bert.save_adapter_fusion(save_path, adapter_fusion_name)
        self.mnetwork.bert.delete_adapter_fusion(adapter_fusion_name)

    def _activate_adapter_runtime(self, adapter_name):
        save_path = os.path.join(self.adapter_cache_path, adapter_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        self.mnetwork.bert.load_adapter(save_path, load_as=adapter_name, set_active=True)

    def _copy_adapter_fusion_runtime(self, new_adapter_fusion_name, adapter_fusion_name):
        save_path = os.path.join(self.adapter_cache_path, adapter_fusion_name)
        assert os.path.exists(save_path)

        self.mnetwork.bert.load_adapter_fusion(save_path, load_as=new_adapter_fusion_name, set_active=True)

    def _create_adapter_fusion_runtime(self, adapter_fusion_name):
        self.mnetwork.bert.add_adapter_fusion(adapter_fusion_name, set_active=True)

    def _copy_adapter_runtime(self, target_adapter, source_adapter):
        save_path = os.path.join(self.adapter_cache_path, source_adapter)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.mnetwork.bert.load_adapter(save_path, load_as=target_adapter, set_active=True)
        
        save_path = os.path.join(self.adapter_cache_path, target_adapter)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.mnetwork.bert.save_adapter(save_path, target_adapter)
    
    def _setup_adv_training(self, config):
        self.adv_teacher = None
        if config.get("adv_train", False):
            self.adv_teacher = SmartPerturbation(
                config["adv_epsilon"],
                config["multi_gpu_on"],
                config["adv_step_size"],
                config["adv_noise_var"],
                config["adv_p_norm"],
                config["adv_k"],
                config["fp16"],
                config["encoder_type"],
                loss_map=self.adv_task_loss_criterion,
                norm_level=config["adv_norm_level"],
            )

    def _get_param_groups(self):
        no_decay = ["bias", "gamma", "beta", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.candidate_params = [n for n, _ in self.network.bert.named_parameters()]
        return optimizer_parameters

    def _setup_optim(self, optimizer_parameters, state_dict=None, num_train_step=-1):
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(optimizer_parameters, self.config['learning_rate'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = AdamaxW(optimizer_parameters,
                                    lr=self.config['learning_rate'],
                                    weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.AdamW(optimizer_parameters,
                                    lr=self.config['learning_rate'],
                                    weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])

        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])


        if state_dict and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        # if self.config["fp16"]:
        #     try:
        #         from apex import amp
        #         global amp
        #     except ImportError:
        #         raise ImportError(
        #             "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
        #         )
        #     model, optimizer = amp.initialize(
        #         self.network, self.optimizer, opt_level=self.config["fp16_opt_level"]
        #     )
        #     self.network = model
        #     self.optimizer = optimizer

        # # set up scheduler
        self.scheduler = None
        scheduler_type = self.config['scheduler_type']
        warmup_steps = self.config['warmup'] * num_train_step
        if scheduler_type == 3:
            from transformers import get_polynomial_decay_schedule_with_warmup
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )
        if scheduler_type == 2:
            from transformers import get_constant_schedule_with_warmup
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps
                )
        elif scheduler_type == 1:
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )
        else:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )


    def _setup_lossmap(self, config):
        task_def_list = config["task_def_list"]
        self.task_loss_criterion = []
        for idx, task_def in enumerate(task_def_list):
            cs = task_def.loss
            lc = LOSS_REGISTRY[cs](name="Loss func of task {}: {}".format(idx, cs))
            self.task_loss_criterion.append(lc)

    def _setup_kd_lossmap(self, config):
        task_def_list = config["task_def_list"]
        self.kd_task_loss_criterion = []
        if config.get("mkd_opt", 0) > 0:
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.kd_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](
                    name="KD Loss func of task {}: {}".format(idx, cs)
                )
                self.kd_task_loss_criterion.append(lc)

    def _setup_adv_lossmap(self, config):
        task_def_list = config["task_def_list"]
        self.adv_task_loss_criterion = []
        if config.get("adv_train", False):
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.adv_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](
                    name="Adv Loss func of task {}: {}".format(idx, cs)
                )
                self.adv_task_loss_criterion.append(lc)

    def _setup_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["init_checkpoint"],
                cache_dir=self.config["transformer_cache"],
            )
        except:
            self.tokenizer = None

    def _to_cuda(self, tensor):
        if tensor is None:
            return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            # y = [e.cuda(non_blocking=True) for e in tensor]
            y = [e.to(self.device) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            # y = tensor.cuda(non_blocking=True)
            y = tensor.to(self.device)
            y.requires_grad = False
        return y

    def compute_loss(self, batch_meta, batch_data):
        self.network.train()
        y = batch_data[batch_meta["label"]]
        y = self._to_cuda(y) if self.config["cuda"] else y
        if batch_meta["task_def"]["task_type"] == TaskType.SeqenceGeneration:
            seq_length = y.size(1)
            y = y.view(-1)

        task_id = batch_meta["task_id"]
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        if "y_token_id" in batch_meta:
            inputs.append(batch_data[batch_meta["y_token_id"]])
        weight = None
        if self.config.get("weighted_on", False):
            if self.config["cuda"]:
                weight = batch_data[batch_meta["factor"]].cuda(non_blocking=True)
            else:
                weight = batch_data[batch_meta["factor"]]

        # fw to get logits
        logits = self.mnetwork(*inputs)

        # compute loss
        loss = 0
        if self.task_loss_criterion[task_id] and (y is not None):
            loss_criterion = self.task_loss_criterion[task_id]
            if (
                isinstance(loss_criterion, RankCeCriterion)
                and batch_meta["pairwise_size"] > 1
            ):
                # reshape the logits for ranking.
                loss = self.task_loss_criterion[task_id](
                    logits,
                    y,
                    weight,
                    ignore_index=-1,
                    pairwise_size=batch_meta["pairwise_size"],
                )
            elif batch_meta["task_def"]["task_type"] == TaskType.SeqenceGeneration:
                weight = (
                    (
                        1.0
                        / torch.sum(
                            (y > -1).float().view(-1, seq_length), 1, keepdim=True
                        )
                    )
                    .repeat(1, seq_length)
                    .view(-1)
                )
                loss = self.task_loss_criterion[task_id](
                    logits, y, weight, ignore_index=-1
                )
            else:
                loss = self.task_loss_criterion[task_id](
                    logits, y, weight, ignore_index=-1
                )

        # compute kd loss
        if self.config.get("mkd_opt", 0) > 0 and ("soft_label" in batch_meta):
            soft_labels = batch_meta["soft_label"]
            soft_labels = (
                self._to_cuda(soft_labels) if self.config["cuda"] else soft_labels
            )
            kd_lc = self.kd_task_loss_criterion[task_id]
            kd_loss = (
                kd_lc(logits, soft_labels, weight, ignore_index=-1) if kd_lc else 0
            )
            loss = loss + kd_loss

        # adv training
        if self.config.get("adv_train", False) and self.adv_teacher:
            # task info
            task_type = batch_meta["task_def"]["task_type"]
            adv_inputs = (
                [self.mnetwork, logits]
                + inputs
                + [task_type, batch_meta.get("pairwise_size", 1)]
            )
            adv_loss, emb_val, eff_perturb = self.adv_teacher.forward(*adv_inputs)
            loss = loss + self.config["adv_alpha"] * adv_loss

        batch_size = batch_data[batch_meta["token_id"]].size(0)
        # rescale loss as dynamic batching
        if self.config["bin_on"]:
            loss = loss * (1.0 * batch_size / self.config["batch_size"])
        if self.config["local_rank"] != -1:
            # print('Rank ', self.config['local_rank'], ' loss ', loss)
            copied_loss = copy.deepcopy(loss.data)
            torch.distributed.all_reduce(copied_loss)
            copied_loss = copied_loss / self.config["world_size"]
            self.train_loss.update(copied_loss.item(), batch_size)
        else:
            self.train_loss.update(loss.item(), batch_size)

        if self.config.get("adv_train", False) and self.adv_teacher:
            if self.config["local_rank"] != -1:
                copied_adv_loss = copy.deepcopy(adv_loss.data)
                torch.distributed.all_reduce(copied_adv_loss)
                copied_adv_loss = copied_adv_loss / self.config["world_size"]
                self.adv_loss.update(copied_adv_loss.item(), batch_size)

                copied_emb_val = copy.deepcopy(emb_val.data)
                torch.distributed.all_reduce(copied_emb_val)
                copied_emb_val = copied_emb_val / self.config["world_size"]
                self.emb_val.update(copied_emb_val.item(), batch_size)

                copied_eff_perturb = copy.deepcopy(eff_perturb.data)
                torch.distributed.all_reduce(copied_eff_perturb)
                copied_eff_perturb = copied_eff_perturb / self.config["world_size"]
                self.eff_perturb.update(copied_eff_perturb.item(), batch_size)
            else:
                self.adv_loss.update(adv_loss.item(), batch_size)
                self.emb_val.update(emb_val.item(), batch_size)
                self.eff_perturb.update(eff_perturb.item(), batch_size)

        # scale loss
        loss = loss / self.config.get("grad_accumulation_step", 1)

        return loss

    def update(self, batch_meta, batch_data):
        loss = self.compute_loss(batch_meta, batch_data)
        # if self.config["fp16"]:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        loss.backward()
        self.local_updates += 1
        if self.local_updates % self.config.get("grad_accumulation_step", 1) == 0:
            if self.config["global_grad_clipping"] > 0:
                # if self.config["fp16"]:
                #     torch.nn.utils.clip_grad_norm_(
                #         amp.master_params(self.optimizer),
                #         self.config["global_grad_clipping"],
                #     )
                # else:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config["global_grad_clipping"]
                )
            self.updates += 1
            # reset number of the grad accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()

    def encode(self, batch_meta, batch_data):
        self.network.eval()
        inputs = batch_data[:3]
        sequence_output = self.network.encode(*inputs)[0]
        return sequence_output

    # TODO: similar as function extract, preserve since it is used by extractor.py
    # will remove after migrating to transformers package
    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def predict(self, batch_meta, batch_data):
        self.network.eval()
        task_id = batch_meta["task_id"]
        task_def = TaskDef.from_dict(batch_meta["task_def"])
        task_type = task_def.task_type
        task_obj = tasks.get_task_obj(task_def)
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        if task_type == TaskType.SeqenceGeneration:
            # y_idx, #3 -> gen
            inputs.append(None)
            inputs.append(3)

        score = self.mnetwork(*inputs)
        if task_obj is not None:
            score, predict = task_obj.test_predict(score)
        elif task_type == TaskType.Ranking:
            score = score.contiguous().view(-1, batch_meta["pairwise_size"])
            assert task_type == TaskType.Ranking
            score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.zeros(score.shape, dtype=int)
            positive = np.argmax(score, axis=1)
            for idx, pos in enumerate(positive):
                predict[idx, pos] = 1
            predict = predict.reshape(-1).tolist()
            score = score.reshape(-1).tolist()
            return score, predict, batch_meta["true_label"]
        elif task_type == TaskType.SeqenceLabeling:
            mask = batch_data[batch_meta["mask"]]
            score = score.contiguous()
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
            valied_lenght = mask.sum(1).tolist()
            final_predict = []
            for idx, p in enumerate(predict):
                final_predict.append(p[: valied_lenght[idx]])
            score = score.reshape(-1).tolist()
            return score, final_predict, batch_meta["label"]
        elif task_type == TaskType.Span or task_type == TaskType.SpanYN:
            predictions = []
            features = []
            for idx, offset in enumerate(batch_meta["offset_mapping"]):
                token_is_max_context = (
                    batch_meta["token_is_max_context"][idx]
                    if batch_meta.get("token_is_max_context", None)
                    else None
                )
                sample_id = batch_meta["uids"][idx]
                if "label" in batch_meta:
                    feature = {
                        "offset_mapping": offset,
                        "token_is_max_context": token_is_max_context,
                        "uid": sample_id,
                        "context": batch_meta["context"][idx],
                        "answer": batch_meta["answer"][idx],
                        "label": batch_meta["label"][idx],
                    }
                else:
                    feature = {
                        "offset_mapping": offset,
                        "token_is_max_context": token_is_max_context,
                        "uid": sample_id,
                        "context": batch_meta["context"][idx],
                        "answer": batch_meta["answer"][idx],
                    }
                if "null_ans_index" in batch_meta:
                    feature["null_ans_index"] = batch_meta["null_ans_index"]
                features.append(feature)
            start, end = score
            start = start.contiguous()
            start = start.data.cpu()
            start = start.numpy().tolist()
            end = end.contiguous()
            end = end.data.cpu()
            end = end.numpy().tolist()
            return (start, end), predictions, features
        elif task_type == TaskType.SeqenceGeneration:
            predicts = self.tokenizer.batch_decode(score, skip_special_tokens=True)
            predictions = {}
            golds = {}
            for idx, predict in enumerate(predicts):
                sample_id = batch_meta["uids"][idx]
                answer = batch_meta["answer"][idx]
                predict = predict.strip()
                if predict == DUMPY_STRING_FOR_EMPTY_ANS:
                    predict = ""
                predictions[sample_id] = predict
                golds[sample_id] = answer
            score = score.contiguous()
            score = score.data.cpu()
            score = score.numpy().tolist()
            return score, predictions, golds
        elif task_type == TaskType.ClozeChoice:
            score = score.contiguous().view(-1)
            score = score.data.cpu()
            score = score.numpy()
            copy_score = score.tolist()
            answers = batch_meta["answer"]
            choices = batch_meta["choice"]
            chunks = batch_meta["pairwise_size"]
            uids = batch_meta["uids"]
            predictions = {}
            golds = {}
            for chunk in chunks:
                uid = uids[0]
                answer = eval(answers[0])
                choice = eval(choices[0])
                answers = answers[chunk:]
                choices = choices[chunk:]
                current_p = score[:chunk]
                score = score[chunk:]
                positive = np.argmax(current_p)
                predict = choice[positive]
                predictions[uid] = predict
                golds[uid] = answer
            return copy_score, predictions, golds            
        else:
            raise ValueError("Unknown task_type: %s" % task_type)
        return score, predict, batch_meta["label"]

    def save(self, filename):
        if isinstance(self.mnetwork, torch.nn.parallel.DistributedDataParallel):
            model = self.mnetwork.module
        else:
            model = self.network

        if not self.adapter:
            # network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
            network_state = dict([(k, v.cpu()) for k, v in model.state_dict().items()])
            params = {
                "state": network_state,
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            }
            torch.save(params, filename)
            logger.info("model saved to {}".format(filename))
        else:
            self.save_all_adapters('/'.join(filename.split('/')[:-1]))

            network_state = dict([(k, v.cpu()) for k, v in model.state_dict().items() if 'bert' not in k])
            params = {
                "state": network_state,
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            }
            torch.save(params, filename)
            logger.info("model saved to {}".format(filename))

    def load(self, checkpoint):
        model_state_dict = torch.load(checkpoint)
        if "state" in model_state_dict:
            self.network.load_state_dict(model_state_dict["state"], strict=False)
        if "optimizer" in model_state_dict:
            self.optimizer.load_state_dict(model_state_dict["optimizer"])
        if "config" in model_state_dict:
            self.config.update(model_state_dict["config"])

        if isinstance(self.mnetwork, torch.nn.parallel.DistributedDataParallel):
            model = self.mnetwork.module
        else:
            model = self.network
        if self.adapter:
            self._load_adapters(model.bert, '/'.join(checkpoint.split('/')[:-1]))

    def cuda(self):
        self.network.cuda()
    
    def _load_adapters(self, model, resume_from_checkpoint):
        adapter_loaded = False
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "adapter_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    model.load_adapter(os.path.join(os.path.join(resume_from_checkpoint, file_name)))
                    adapter_loaded = True
        return adapter_loaded


    def save_all_adapters(self, output_dir=None):
        import json
        # If we are executing this function, we are the process zero, so we don't check for that.
        # output_dir = output_dir if output_dir is not None else self.output_dir
        # os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving adapter checkpoint to {output_dir}")

        if not self.train_adapter_fusion:
            activate_adapters = []
            for layer in self.laryerwise_candidate_adapter.keys():
                for target_task in self.laryerwise_candidate_adapter[layer].keys():
                    activate_adapters.append(self.laryerwise_candidate_adapter[layer][target_task])
            
            activate_adapters = list(set(activate_adapters))

            current_activate_adapters = list(self.current_active_adapters.values())

            for adapter in activate_adapters:
                if adapter in current_activate_adapters:
                    self.mnetwork.bert.save_adapter(os.path.join(output_dir, adapter), adapter)
                else:
                    adapter_path = f'{self.adapter_cache_path}/{adapter}'
                    os.system(f'cp -rf {adapter_path} {output_dir}')
        else:
            activate_adapter_fusions = []
            for layer in self.laryerwise_candidate_adapter.keys():
                layer_activate_adapters = list(set(list(self.laryerwise_candidate_adapter[layer].values())))

                if len(layer_activate_adapters) == 1:
                    adapter = layer_activate_adapters[0]
                    self.mnetwork.bert.save_adapter(os.path.join(output_dir, adapter), adapter)
                else:
                    assert self.laryerwise_fusion_adapters[layer][0]

                    adapter_fusion = self.laryerwise_fusion_adapters[layer][1]
                    # print('>>> ADAPTER FUSION', adapter_fusion)
                    activate_adapter_fusions.append(adapter_fusion)
                    self.mnetwork.bert.save_adapter_fusion(os.path.join(output_dir, adapter_fusion), adapter_fusion)
                    for adapter in adapter_fusion.split(','):
                        self.mnetwork.bert.save_adapter(os.path.join(output_dir, adapter), adapter)
            json.dump(activate_adapter_fusions, open(os.path.join(output_dir, 'activate_adapter_fusions.json'), 'w'), indent=4)
            
        json.dump(self.laryerwise_candidate_adapter, open(os.path.join(output_dir, 'adapter_structure.json'), 'w'), indent=4)