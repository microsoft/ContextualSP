import absl
import nltk
import numpy
import six

import datasets

import pdb

_CITATION = ""

_DESCRIPTION = ""

_KWARGS_DESCRIPTION = ""


def simple_accuracy(preds, labels):
    correct_list = [1. if pred == label else 0. for (pred, label) in zip(preds, labels)]
    return sum(correct_list) / len(correct_list)


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class seq_acc(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    def _compute(self, predictions, references):
        accuracy = simple_accuracy(predictions, references)
        result = {'accuracy': accuracy}

        return result
