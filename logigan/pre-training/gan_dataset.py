from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy
import copy
from dataclasses import dataclass
InputDataClass = NewType("InputDataClass", Any)

@dataclass
class DataCollatorForGAN:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_instance_num: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        new_features = []
        batch_size = self.max_instance_num
        for ins_feature in features:
            if len(new_features)>=batch_size:
                break
            tmp_features = []
            gold_idx = len(ins_feature)

            for i in range(len(ins_feature['labels'])):
                    tmp_features.append(copy.deepcopy(ins_feature))
                    tmp_features[-1]['labels'] = ins_feature['labels'][i]
                    if 'is_gold' in ins_feature.keys():
                        tmp_features[-1]['is_gold'] = ins_feature['is_gold'][i]
                        gold_idx = ins_feature['is_gold'].index(1)

                    if 'ver_prob' in ins_feature.keys():
                        tmp_features[-1]['ver_prob'] = ins_feature['ver_prob'][i]
            new_features.append(tmp_features[gold_idx])
            rest_num = min(max(0,batch_size-len(new_features)),len(tmp_features)-1)
            if rest_num!=0:
                fake_ids = np.random.choice(np.arange(len(tmp_features)), rest_num+1,
                                            replace=False) if rest_num + 1 < len(tmp_features) else np.arange(len(tmp_features))
                selected_negs = [tmp_features[id] for id in fake_ids if id!=gold_idx][:rest_num]
                new_features.extend(selected_negs)
        # print(new_features)
        # print(len(new_features))
        # exit()
        labels = [feature["labels"] for feature in new_features] if "labels" in new_features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in new_features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        new_features = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=new_features["labels"])
            new_features["decoder_input_ids"] = decoder_input_ids

        return new_features
