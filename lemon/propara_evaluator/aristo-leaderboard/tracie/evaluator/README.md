## TRACIE Evaluator

This script evaluates NLI predictions against correct inferences and produces 4 accuracy scores described below, and can be used to check that outputs produced for the leaderboard are well formed. 

## Example

```sh
% python3 evaluator/evaluator.py --question_answers data/train_uniform.jsonl --predictions data/predictions.jsonl --output metrics.json 

% cat metrics.json
{"total_acc": 0.5, "start_acc": 0.5, "end_acc": 0.5, "story_em": 0.0}
```

This uses a dummy prediction file called `predictions.jsonl` which predicts entailments for each example in `train_uniform.jsonl`:

```
{"id":"tracie-train-uniform-0000","label":"entailment"}
{"id":"tracie-train-uniform-0001","label":"entailment"}
{"id":"tracie-train-uniform-0002","label":"entailment"}
{"id":"tracie-train-uniform-0003","label":"entailment"}
{"id":"tracie-train-uniform-0004","label":"entailment"}
...
```

## Output metrics

A json file called `metrics.json` will be produced containing the following accuracy scores:

```json
{"train_type": "train_iid", "total_acc": 0.5, "start_acc": 0.5, "end_acc": 0.5, "story_em": 0.0}
```

In this file, here is what the fields mean:

* `total_acc` is the overall accuracy
* `start_acc` is the accuracy of the subset of problems involving event `start` questions
* `end_acc` is the subset involving end point questions
* `story_em` is the accuracy of getting all questions correct per story
