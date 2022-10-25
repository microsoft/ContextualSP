from transformers import Seq2SeqTrainer
from typing import Dict, List, Optional
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union,NamedTuple
from transformers import Seq2SeqTrainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
import torch.nn as nn
import collections
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    denumpify_detensorize,
    EvalLoopOutput,
    EvalPrediction)
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
logger = logging.getLogger(__name__)
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]

class GenTrainer(Seq2SeqTrainer):
    def __init__(self, *args, num_return_seq=4,num_beams=4,gan_alpha=0.8,**kwargs):
        super().__init__(*args, **kwargs)
        self.num_return_seq = num_return_seq
        self._num_beams = num_beams
        # self.gan_alpha = gan_alpha
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        # print(f'The coefficient for teacher loss: {gan_alpha}')

    def compute_loss(self, model, inputs):
        outputs = model(**inputs, output_hidden_states=False)
        return outputs.loss
    '''
    def compute_loss(self, model, inputs):
        assert "labels" in inputs, """labels is required to compute loss"""
        is_gold = inputs.pop("is_gold")  # Should be torch.LongTensor
        ver_prob = inputs.pop("ver_prob")  # Should be torch.FloatTensor
        outputs = model(**inputs, output_hidden_states=False)
        entropy = outputs.loss  # All loss
        teacher_forcing_loss = is_gold * entropy  # We only consider teacher forcing loss when is_gold=True
        ver_prob = ((1 - is_gold) * ver_prob)[1:]
        gen_score = 1 - ((1 - is_gold) * entropy)[1:]
        ver_prob_norm = torch.softmax(ver_prob,dim=-1)
        gen_score_norm = torch.softmax(gen_score,dim=-1)
        gan_loss = self.kl_loss(ver_prob_norm,gen_score_norm)
        # gen_score = gen_score
        # ver_prob_rank = torch.argsort(ver_prob, dim=-1, descending=True).float().unsqueeze(0)
        # gen_score_rank = torch.argsort(gen_score, dim=-1, descending=True).float().unsqueeze(0)
        # gan_loss = 1 - torch.cosine_similarity(ver_prob_rank, gen_score_rank,
        #                                        dim=-1)  # torch.cosine_embedding_loss(ver_prob_rank,gen_score_rank,target=torch.Tensor(1).cuda())
        gan_alpha = self.gan_alpha  # model.module.gan_alpha # self.gan_alpha
        # print(gan_alpha)
        ALPHA = gan_alpha
        BETA = 1 - gan_alpha
        loss = (ALPHA * teacher_forcing_loss).sum() + (BETA * gan_loss).sum()

        # print(f"is_gold: {is_gold}")
        # print(
        #     f"Verifier predict probability (ver_prob): {ver_prob}({ver_prob_norm}),(score prob) {gen_score} ({gen_score_norm})")
        # print(f"Cross entropy loss: {entropy}")
        # print(f"teacher_forcing_loss: {teacher_forcing_loss}")
        # print(f"gan_loss: {gan_loss}")
        # print(f"Total loss: {loss}")

        # exit()
        return loss
    '''
    '''
    def compute_loss(self, model, inputs):
        assert "labels" in inputs,  """labels is required to compute loss"""
        is_gold = inputs.pop("is_gold")  # Should be torch.LongTensor
        ver_prob = inputs.pop("ver_prob") # Should be torch.FloatTensor
        outputs = model(**inputs, output_hidden_states=False)
        entropy = outputs.loss # All loss
        teacher_forcing_loss = is_gold * entropy # We only consider teacher forcing loss when is_gold=True
        ver_prob = ((1-is_gold)*ver_prob)[1:]
        gen_score = 1-((1-is_gold)*entropy)[1:]
        gen_score = gen_score
        ver_prob_rank = torch.argsort(ver_prob,dim=-1,descending=True).float().unsqueeze(0)
        gen_score_rank = torch.argsort(gen_score,dim=-1,descending=True).float().unsqueeze(0)
        gan_loss = 1-torch.cosine_similarity(ver_prob_rank,gen_score_rank,dim=-1)#torch.cosine_embedding_loss(ver_prob_rank,gen_score_rank,target=torch.Tensor(1).cuda())
        gan_alpha = self.gan_alpha#model.module.gan_alpha # self.gan_alpha
        # print(gan_alpha)
        ALPHA = gan_alpha
        BETA = 1 - gan_alpha
        loss = (ALPHA * teacher_forcing_loss).sum() + (BETA * gan_loss).sum()
        
        print(f"is_gold: {is_gold}")
        print(f"Verifier predict probability (ver_prob): {ver_prob}({ver_prob_rank}),(score prob) {gen_score} ({gen_score_rank})")
        print(f"Cross entropy loss: {entropy}")
        print(f"teacher_forcing_loss: {teacher_forcing_loss}")
        print(f"gan_loss: {gan_loss}")
        print(f"Total loss: {loss}")
        
        # exit()
        return loss
    '''
    '''
    def compute_loss(self, model, inputs):
        assert "labels" in inputs,  """labels is required to compute loss"""
        is_gold = inputs.pop("is_gold")  # Should be torch.LongTensor
        ver_prob = inputs.pop("ver_prob") # Should be torch.FloatTensor
        outputs = model(**inputs, output_hidden_states=False)
        entropy = outputs.loss # All loss
        teacher_forcing_loss = is_gold * entropy # We only consider teacher forcing loss when is_gold=True
        normalized_ver_prob = (1 - is_gold) * ver_prob
        gan_loss = normalized_ver_prob * entropy # We only add GAN-loss for is_gold=False
        ALPHA = 1
        BETA = 1
        loss = (ALPHA * teacher_forcing_loss + BETA * gan_loss).sum()
        
        print(f"is_gold: {is_gold}")
        print(f"Verifier predict probability (ver_prob): {ver_prob}")
        print(f"Cross entropy loss: {entropy}")
        print(f"teacher_forcing_loss: {teacher_forcing_loss}")
        print(f"gan_loss: {gan_loss}")
        print(f"Total loss: {loss}")
        
        # exit()
        return loss
    '''
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": False,  # True if is_deepspeed_zero3_enabled() else False,
            "num_return_sequences": self.num_return_seq if self.num_return_seq else 10
        }
        # print(gen_kwargs['num_beams'],gen_kwargs['num_return_sequences'])

        if self.tokenizer is not None:
            generation_inputs = {k: v for k, v in inputs.items() if k in self.tokenizer.model_input_names}
            # very ugly hack to make it work
            generation_inputs["input_ids"] = generation_inputs.pop(self.tokenizer.model_input_names[0])
        else:
            generation_inputs = inputs["input_ids"]

        generated_tokens = self.model.generate(
            **generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        # print(all_generated_tokens.shape)
        # for i in range(all_generated_tokens.size(1)):
        #     generated_tokens = all_generated_tokens[:,i,:]
        #     if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
        #         all_generated_tokens[:,i,:] = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        generated_tokens = generated_tokens.view(-1, gen_kwargs["num_return_sequences"], gen_kwargs["max_length"])
        # print(generated_tokens.size())
        return (loss, generated_tokens, labels)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            preds_for_eval = [preds[0] for preds in all_preds]
            # print([len(pred) for pred in preds_for_eval])
            # print(len(all_labels))
            metrics = self.compute_metrics(EvalPrediction(predictions=preds_for_eval, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
