# QASC

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.
* `data` have example prediction files

## Example usage

To evaluate your predictions against the train or dev datasets, run either of these and look at the resulting metrics.json file:

```
% python3 evaluator/evaluator.py -qa data/train.jsonl -p /path/to/your/train/predictions.csv -o metrics.json 
% python3 evaluator/evaluator.py -qa data/dev.jsonl -p /path/to/your/dev/predictions.csv -o metrics.json 
```

For example, to evaluate dummy predictions (every question is predicted to be `A`) against the train dataset, run this:

```
% python3 evaluator/evaluator.py -qa data/train.jsonl -p data/train-predictions.csv -o metrics.json 

% cat metrics.json
{"accuracy": 0.12417014998770592}
```

For usage of the evaluator, see the [evaluator README](evaluator/).
