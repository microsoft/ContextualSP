This directory contains the training and test files for evaluating predictions,
and a sample prediction file.

The file `train_uniform.jsonl` is the main training data to use for
leaderboard entries (please note that in our paper, we also experiment
with training on an `iid` set (not included here) _**which is not
allowed when submiting to the leaderboard**). 

The file `test.jsonl` has the test questions, without labels.

Each example in these files looks like the following:

```json
{
  "query": "event: Tom's teeth are crooked ends before he has braces on for a while",
  "story": "Tom needed to get braces. He was afraid of them. The dentist assured him everything would be fine. Tom had them on for a while. Once removed he felt it was worth it.",
  "label": "contradiction"
}
```

and consists of three fields:

* `query` (or hypothesis)
* `story` (or premise)
* `label` (the inference label; this is absent in `test.jsonl`)

The file `predictions.jsonl` shows an example prediction file for the `uniform`
training split that can be evaluated against `train_uniform.jsonl`.
