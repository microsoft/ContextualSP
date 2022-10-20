## ProPara datasets

This directory contains dev, train and test datasets. 
  
  * [dev](dev/) contains the dev dataset for developing your predictor
  * [train](train/) contains the training dataset for evaluating your predictor during development
  * [test](test/) contains the test dataset for evaluation on the [ProPara Leaderboard](https://leaderboard.allenai.org/).

Each subdirectory contains `answers.tsv` and `dummy-predictions.tsv` files. In
addition each has a `sentences.tsv` file as a convenience to discover the
process paragraphs for each process id.
