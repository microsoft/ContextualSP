# SciTail

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.

## Example usage

To evaluate dummy predictions (every pair of sentences is predicted to entail) against the SciTail dataset, run this:

```
% python3 evaluator/evaluator.py -a data/test/answers.jsonl -p data/test/dummy-predictions.csv
accuracy: 0.39604891815616183
```

Replace `data/test/dummy-predictions.csv` with your predictions to compute your test score.

You can also evaluate predictions against the Dev set by running:

```
% python3 evaluator/evaluator.py -a data/dev/answers.jsonl -p data/dev/dummy-predictions.csv
accuracy: 0.5038343558282209
```

Replace `data/dev/dummy-predictions.csv` with your predictions to compute your dev score. 
