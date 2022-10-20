#!/usr/bin/env python3

import csv
from typing import *
import logging
import sys
import json

EXIT_STATUS_ANSWERS_MALFORMED = 1
EXIT_STATUS_PREDICTIONS_MALFORMED = 2
EXIT_STATUS_PREDICTIONS_EXTRA = 3
EXIT_STATUS_PREDICTION_MISSING = 4
VALID_PREDICTION_VALUES = ['E', 'N']


def calculate_accuracy(answers: Dict[str, str], predictions: Dict[str, str]) -> float:
    score = 0.0

    for entailment_pair_id, answer in answers.items():
        try:
            predictions_for_q = predictions[entailment_pair_id]
        except KeyError:
            logging.error("Missing prediction for entailment pair '%s'.", entailment_pair_id)
            sys.exit(EXIT_STATUS_PREDICTION_MISSING)

        if answer in predictions_for_q:
            score += 1

        del predictions[entailment_pair_id]

    if len(predictions) > 0:
        logging.error("Found %d extra predictions, for example: %s", len(predictions),
                      ", ".join(list(predictions.keys())[:3]))
        sys.exit(EXIT_STATUS_PREDICTIONS_EXTRA)

    return score / len(answers)


def read_answers(filename: str) -> Dict[str, str]:
    answers = {}  # type: Dict[str, str]

    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            try:
                record = json.loads(line)
            except ValueError as e:
                logging.error("Error while reading file %s: %s", filename, e)
                sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

            entailment_pair_id = record["id"]
            answer = record["gold_label"]

            if entailment_pair_id in answers:
                logging.error("Key %s repeated in %s", entailment_pair_id, filename)
                sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)
            answers[entailment_pair_id] = answer

    if len(answers) == 0:
        logging.error("No answers found in file %s", filename)
        sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

    return answers


def read_predictions(filename: str) -> Dict[str, str]:
    predictions = {}  # type: Dict[str, str]

    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                try:
                    entailment_pair_id = row[0]
                    prediction = row[1]
                except IndexError as e:
                    logging.error("Error reading value from CSV file %s on line %d: %s", filename, reader.line_num, e)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                if entailment_pair_id in predictions:
                    logging.error("Key %s repeated in file %s on line %d", entailment_pair_id, filename,
                                  reader.line_num)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                if entailment_pair_id == "":
                    logging.error("Key is empty in file %s on line %d", filename, reader.line_num)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                # prediction cannot be empty string
                if prediction == "":
                    logging.error("Key %s has empty string for prediction in file %s on line %d",
                                  entailment_pair_id, filename, reader.line_num)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                # predictions must be part of the controlled vocabulary
                if prediction not in VALID_PREDICTION_VALUES:
                    logging.error("Key %s has invalid prediction in file %s on line %d",
                                  entailment_pair_id, filename, reader.line_num)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                predictions[entailment_pair_id] = prediction

        except csv.Error as e:
            logging.error('file %s, line %d: %s', filename, reader.line_num, e)
            sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

    return predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for SciTail sentence pairs.')

    parser.add_argument(
        '--answers', '-a',
        help='Filename of the answers to read. Expects a JSONL file with documents that have fields "id" and '
             '"gold_label".',
        required=True)
    parser.add_argument(
        '--predictions', '-p',
        help="Filename of the leaderboard predictions, in CSV format.",
        required=True)
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file.')

    args = parser.parse_args()

    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    accuracy = calculate_accuracy(answers, predictions)

    if args.output:
        print("Writing results to file: %s" % args.output)
        with open(args.output, "wt", encoding="UTF-8") as output:
            output.write(json.dumps({"accuracy": accuracy}))
    else:
        print("accuracy:", accuracy)


if __name__ == '__main__':
    main()
