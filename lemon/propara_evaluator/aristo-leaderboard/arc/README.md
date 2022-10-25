# AI2 Reasoning Challenge

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.
* [data-easy](data-easy/) and [data-challege](data-challenge/) have the files (and scripts to generate them) used for evaluating Leaderboard predictions.

## Example usage

To evaluate dummy predictions (every question is predicted to be `A`) against the easy dataset, run this:

```
% python3 evaluator/evaluator.py -qa data-easy/question-answers.jsonl -p data-easy/dummy-predictions.csv -o metrics.json 

% cat metrics.json
{"accuracy": 0.2398989898989899}
```
