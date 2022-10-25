# eQASC

This directory has code and data for the eQASC evaluator, as described in the EMNLP 2020 paper [Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering](https://www.semanticscholar.org/paper/Learning-to-Explain%3A-Datasets-and-Models-for-Valid-Jhamtani-Clark/ebaebfefec9d5c21a4559a1a038743bd437d2f01).

* [code/](code/) holds the evaluator
* [data/](data/) holds the labels used by the evaluator

## Example usage

To evaluate your prediction file (located at /tmp/my_predictions_test.jsonl) against the
test dataset, run this and look at the scores in the file /tmp/metrics.json:

```
cd code
docker build -t eqasc-evaluator .
docker run \
  -e PYTHONPATH=. \
  -e PYTHONUNBUFFERED=yes \
  -v /tmp/my_predictions_test.jsonl:/predictions.jsonl:ro \
  -v $PWD/../data/chainid_to_label_test.json:/labels.json:ro \
  -v /tmp:/output:rw \
  --entrypoint python \
  eqasc-evaluator \
  allennlp_reasoning_explainqa/evaluator/evaluator.py \
  /predictions.jsonl \
  /labels.json \
  /output/metrics.json
```

To confirm that the evaluator is working on correct inputs, you can use [dummy
prediction files](data/). To to do, replace `/tmp/my_predictions_test.jsonl` above
with `$PWD/../data/dummy_predictions_test.jsonl`.

You'll find more details about the evaluator in the [code/](code/) directory.

## Reference

Please cite the work like this:

```
@inproceedings{jhamtani2020grc,
  title={Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering},
  author={Jhamtani, Harsh and Clark, Peter},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}
```
