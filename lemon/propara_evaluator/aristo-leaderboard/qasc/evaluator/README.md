## QASC Evaluator

This script evaluates predictions for multiple-choice questions against correct answers and produces an accuracy score.

## Example

```bash
% python3 evaluator.py -qa questions.jsonl -p predictions.csv -o metrics.json

% cat metrics.json
{"accuracy": 0.85}
```

## Usage

The script takes two input files and produces one output file.

### Input question-answers

A question-answers file has question ids and answers in JSONL format. For example:

```bash
% cat questions.jsonl
{ "id": "question1", "answerKey": "C" }
{ "id": "question2", "answerKey": "B" }
{ "id": "question3", "answerKey": "C" }
{ "id": "question4", "answerKey": "D" }
{ "id": "question5", "answerKey": "D" }
```

(Attributes besides `id` and `answerKey` in each object are ignored.)

### Input predictions

A predictions file that has predictions in CSV format. For example:

```bash
% cat predictions.csv
question1,A;B;C;D
question2,B
question3,C
question4,D
question5,D
```

### Output metrics

A JSON file that has an accuracy score in the range 0.0 to 1.0. For example:

```bash
% cat metrics.json 
{"accuracy": 0.85}
```

## Development

### Unit tests

Run unit tests with `python3 test_evaluator.py`.

### Docker

Ultimately this evaluator is run in a Docker container. To test that it works there, run `test.sh`.


