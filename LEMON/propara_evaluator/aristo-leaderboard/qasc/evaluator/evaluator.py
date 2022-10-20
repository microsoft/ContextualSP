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


def calculate_accuracy(question_answers: Dict[str, str], predictions: Dict[str, List[str]]) -> float:
    score = 0.0

    for question_id, answer in question_answers.items():
        try:
            predictions_for_q = predictions[question_id]
        except KeyError:
            logging.error("Missing prediction for question '%s'.", question_id)
            sys.exit(EXIT_STATUS_PREDICTION_MISSING)

        if answer in predictions_for_q:
            score += 1.0 / len(predictions_for_q)

        del predictions[question_id]

    if len(predictions) > 0:
        logging.error("Found %d extra predictions, for example: %s", len(predictions),
                      ", ".join(list(predictions.keys())[:3]))
        sys.exit(EXIT_STATUS_PREDICTIONS_EXTRA)

    return score / len(question_answers)


def read_answers(filename: str) -> Dict[str, str]:
    answers = {}

    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            try:
                record = json.loads(line)
            except ValueError as e:
                logging.error("Error while reading file %s: %s", filename, e)
                sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

            question_id = record["id"]
            answer = record["answerKey"]

            if question_id in answers:
                logging.error("Key %s repeated in %s", question_id, filename)
                sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)
            answers[question_id] = answer

    if len(answers) == 0:
        logging.error("No answers found in file %s", filename)
        sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

    return answers


def read_predictions(filename: str) -> Dict[str, List[str]]:
    predictions = {}

    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                try:
                    question_id = row[0]
                    prediction_raw = row[1]
                except IndexError as e:
                    logging.error("Error reading value from CSV file %s on line %d: %s", filename, reader.line_num, e)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                if question_id in predictions:
                    logging.error("Key %s repeated in file %s on line %d", question_id, filename, reader.line_num)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                if question_id == "":
                    logging.error("Key is empty in file %s on line %d", filename, reader.line_num)
                    sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

                prediction = prediction_raw.split(";")
                # prediction labels cannot be empty strings
                for p in prediction:
                    if p == "":
                        logging.error("Key %s has empty labels for prediction in file %s on line %d",
                                      question_id, filename, reader.line_num)
                        sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)
                predictions[question_id] = prediction

        except csv.Error as e:
            logging.error('file %s, line %d: %s', filename, reader.line_num, e)
            sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

    return predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for questions.')

    parser.add_argument(
        '--question-answers', '-qa',
        help='Filename of the question answers to read. Expects a JSONL file with documents that have field "id" and "answerKey".',
        required=True)
    parser.add_argument(
        '--predictions', '-p',
        help="Filename of the leaderboard predictions, in CSV format.",
        required=True)
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file.',
        required=True)

    args = parser.parse_args()

    question_answers = read_answers(args.question_answers)
    predictions = read_predictions(args.predictions)
    accuracy = calculate_accuracy(question_answers, predictions)

    with open(args.output, "wt", encoding="UTF-8") as output:
        output.write(json.dumps({"accuracy": accuracy}))


if __name__ == '__main__':
    main()
