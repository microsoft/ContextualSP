The file [test-predictions.csv](test-predictions.csv) is a valid example prediction file that can be submitted to the [QASC Leaderboard](https://leaderboard.allenai.org/). This is a prediction that every question's correct answer is the choice `A`, and scores about 13% correct. This file shows the submission format without revealing the correct answers.

The files [train-predictions.csv](train-predictions.csv) and [dev-predictions.csv](dev-predictions.csv) show similarly random answers (all predictions are for answer choice `A`) for the train and dev datasets.

The files [train.jsonl](train.jsonl) and [dev.jsonl](dev.jsonl) have the correct answers to the train and dev datasets. These can be used for improving the performance of your predictor before it predicts answers to the hidden test questions.
