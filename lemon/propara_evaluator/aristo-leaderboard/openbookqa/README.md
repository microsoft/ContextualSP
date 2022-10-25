# OpenBookQA

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.
* `data` have the files (and scripts to generate them) used for evaluating Leaderboard predictions.

## Example usage

To evaluate dummy predictions (every question is predicted to be `A`) against the dataset, run this:

```
% python3 evaluator/evaluator.py -qa data/question-answers.jsonl -p data/dummy-predictions.csv -o metrics.json 

% cat metrics.json
{"accuracy": 0.276}
```

