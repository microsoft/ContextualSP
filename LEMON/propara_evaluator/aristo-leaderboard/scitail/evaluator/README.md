## SciTail Evaluator

This script evaluates predictions on the SciTail dataset and produces an accuracy score.

## Example

```bash
% python3 evaluator.py -a answers.jsonl -p predictions.csv -o metrics.json

% cat metrics.json
{"accuracy": 0.8}
```

## Usage

The script takes two input files and produces one output file.

### Input answers

A file has id and gold labels in JSONL format. For example:

```bash
% cat answers.jsonl
{ "id": "P1", "gold_label": "E" }
{ "id": "P2", "gold_label": "E" }
{ "id": "P3", "gold_label": "N" }
{ "id": "P4", "gold_label": "N" }
{ "id": "P5", "gold_label": "N" }
```

(Attributes besides `id` and `gold_label` in each object are ignored.)

### Input predictions

A predictions file that has predictions in CSV format. For example:

```bash
% cat predictions.csv
P1,E
P2,N
P3,N
P4,N
P5,N
```

### Output metrics

A JSON file that has an accuracy score in the range 0.0 to 1.0. For example:

```bash
% cat metrics.json 
{"accuracy": 0.8}
```

## Development

### Unit tests

Run unit tests with `python3 test_evaluator.py`.

### Docker

Ultimately this evaluator is run in a Docker container. To test that it works there, run `test.sh`.
