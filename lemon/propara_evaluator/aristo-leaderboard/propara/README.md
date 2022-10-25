# ProPara

* [evaluator](evaluator/) is the program used by the [ProPara Leaderboard](https://leaderboard.allenai.org/) to evaluate submitted predictions.
* [data](data/) contains dev, train and test datasets

## Example usage

To evaluate dummy predictions against the ProPara test dataset, run this:

```
% export PYTHONPATH=.
% evaluator/evaluator.py -p data/test/dummy-predictions.tsv -a data/test/answers.tsv
=================================================
Question     Avg. Precision  Avg. Recall  Avg. F1
-------------------------------------------------
Inputs                1.000        0.241    0.388
Outputs               1.000        0.130    0.230
Conversions           1.000        0.185    0.312
Moves                 1.000        0.222    0.363
-------------------------------------------------
Overall Precision 1.000                          
Overall Recall    0.195                          
Overall F1        0.326                          
=================================================

Evaluated 54 predictions against 54 answers.
```

Replace `data/test/dummy-predictions.tsv` with your predictions to compute your test score.

You can also evaluate predictions against the Dev and Training sets by running:

```
% evaluator/evaluator.py -p data/dev/dummy-predictions.tsv -a data/dev/answers.tsv
% evaluator/evaluator.py -p data/train/dummy-predictions.tsv -a data/train/answers.tsv
```

See the [evaluator documenation](evaluator/) for details about the prediction file format and scoring process.
