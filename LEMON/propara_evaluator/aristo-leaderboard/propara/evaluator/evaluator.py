#!/usr/bin/env python3

import argparse
import json
from typing import Dict

from evaluation import Evaluation
from process import sentences_from_sentences_file, ActionFile
from scoring import QuestionScores
from errors import corrupted_action_file, corrupted_sentences_file


def main(answers_file: str, predictions_file: str, output_file: str, diagnostics_file: str, sentences_file: str):
    # create diagnostics file if needed
    diagnostics = None
    sentences = None
    if diagnostics_file:
        diagnostics = open(diagnostics_file, mode='w')
        print(f"Writing diagnostics to file {diagnostics_file}")
        diagnostics.write(
            f"Diagnostics of evaluation of predictions in {predictions_file} against answers in {answers_file}\n")
        diagnostics.write("\n")
        if sentences_file:
            sentences = sentences_from_sentences_file(sentences_file)

    # Step 1 and 2. Read and summarize answers and predictions
    predictions = ActionFile.from_file(predictions_file)
    answers = ActionFile.from_file(answers_file)

    # Abort if there are differences
    diff_report = answers.diff_participants(predictions)
    if diff_report:
        print(f"Participants in predictions file {predictions_file} are not exact matches to participants")
        print(f"in {answers_file}. Detailed report:")
        print()
        print("\n".join(diff_report))
        print()
        corrupted_action_file(
            filename=predictions_file,
            details=f"Some participants are missing or unexpected."
        )

    predictions_summary = predictions.summarize()
    answers_summary = answers.summarize()

    # Step 3. Calculate per-process scores
    scores_by_process = dict()  # type: Dict[int, QuestionScores]
    for process_id, answer_summary in answers_summary.items():
        if process_id not in predictions_summary:
            corrupted_action_file(
                filename=predictions_file,
                details=f"Prediction for process_id {answer_summary.process_id} is missing."
            )

        prediction_summary = predictions_summary[process_id]

        score = QuestionScores.from_summaries(answer_summary, prediction_summary)
        scores_by_process[process_id] = score

        if diagnostics:
            diag_struct = {
                "prediction_summary": prediction_summary.diagnostics(),
                "answer_summary": answer_summary.diagnostics(),
                "score": {
                    "process_id": process_id,
                    "inputs": score.inputs.diagnostics(),
                    "outputs": score.outputs.diagnostics(),
                    "conversions": score.conversions.diagnostics(),
                    "moves": score.moves.diagnostics(),
                }
            }

            if sentences:
                if process_id not in sentences:
                    corrupted_sentences_file(
                        filename=sentences_file,
                        details=f"Sentences for process {process_id} not found."
                    )
                sentences_for_diag = []
                for i, text in enumerate(sentences[process_id]):
                    sentences_for_diag.append({
                        "step_number": 1 + i,
                        "text": text,
                    })
                diag_struct["sentences"] = sentences_for_diag  # type: ignore

            diagnostics.write(json.dumps(diag_struct, indent=4))
            diagnostics.write("\n")

    # Step 4. Calculate a final evaluation
    evaluation = Evaluation(scores_by_process)

    # Step 5. Print a report and generate output file
    report(evaluation, len(predictions_summary), len(answers_summary))

    overall_scores = {
        "precision": round(evaluation.overall.precision, 3),
        "recall": round(evaluation.overall.recall, 3),
        "f1": round(evaluation.overall.F1(), 3)
    }

    if output_file:
        print("Writing results to file: %s" % output_file)
        with open(output_file, "wt", encoding="UTF-8") as output:
            output.write(json.dumps(overall_scores))

    if diagnostics:
        diag_struct = {"overall_scores": overall_scores}
        diagnostics.write(json.dumps(diag_struct, indent=4))
        diagnostics.write("\n")

    # close diagnostics file
    if diagnostics:
        diagnostics.close()


def report(e: Evaluation, num_predictions: int, num_answers: int):
    i = e.inputs
    o = e.outputs
    c = e.conversions
    m = e.moves
    overall = e.overall
    print("=================================================")
    print("Question     Avg. Precision  Avg. Recall  Avg. F1")
    print("-------------------------------------------------")
    print("Inputs                %4.3f        %4.3f    %4.3f" % (i.precision, i.recall, i.F1()))
    print("Outputs               %4.3f        %4.3f    %4.3f" % (o.precision, o.recall, o.F1()))
    print("Conversions           %4.3f        %4.3f    %4.3f" % (c.precision, c.recall, c.F1()))
    print("Moves                 %4.3f        %4.3f    %4.3f" % (m.precision, m.recall, m.F1()))
    print("-------------------------------------------------")
    print("Overall Precision %4.3f                          " % overall.precision)
    print("Overall Recall    %4.3f                          " % overall.recall)
    print("Overall F1        %4.3f                          " % overall.F1())
    print("=================================================")
    print()
    print(f"Evaluated {num_predictions} predictions against {num_answers} answers.")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator for ProPara Leaderboard')

    parser.add_argument('--predictions', '-p',
                        action='store',
                        dest='predictions_file',
                        required=True,
                        help='Path to file with predictions')

    parser.add_argument('--answers', '-a',
                        action='store',
                        dest='answers_file',
                        required=True,
                        help='Path to file with answers')
    parser.add_argument('--output', '-o',
                        action='store',
                        dest='output_file',
                        help='Output results to this file.')

    parser.add_argument('--diagnostics', '-d',
                        action='store',
                        dest='diagnostics_file',
                        help='Write diagnostics to this file.')

    parser.add_argument('--sentences', '-s',
                        action='store',
                        dest='sentences_file',
                        help='Path to file with sentences.')

    args = parser.parse_args()

    main(args.answers_file, args.predictions_file, args.output_file, args.diagnostics_file, args.sentences_file)
