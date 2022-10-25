import json
import sys


def evaluate(answer_file, prediction_file):
    answer_by_id = {}
    for line in open(answer_file).readlines():
        struct = json.loads(line)
        answer_by_id[struct["id"]] = struct

    prediction_by_id = {}
    for line in open(prediction_file).readlines():
        struct = json.loads(line)
        prediction_by_id[struct["id"]] = struct

    answer_count = len(answer_by_id)
    prediction_count = len(prediction_by_id)
    if answer_count != prediction_count:
        print(
            f"Prediction count ({prediction_count}) doesn't match answer count ({answer_count})"
        )
        sys.exit(1)

    total = 0
    correct = 0
    total_start = 0
    correct_start = 0
    total_end = 0
    correct_end = 0
    story_prediction_map = {}

    for answer in answer_by_id.values():
        answer_id = answer["id"]
        prediction = prediction_by_id.get(answer_id, None)
        if not prediction:
            print(f"Prediction for id {answer_id} missing")
            sys.exit(1)

        hypothesis = answer["query"]
        story = answer["story"]
        answer_label = answer["label"]
        prediction_label = prediction["label"]

        if story not in story_prediction_map:
            story_prediction_map[story] = []

        total += 1
        if answer_label == prediction_label:
            correct += 1
            story_prediction_map[story].append(True)
        else:
            story_prediction_map[story].append(False)

        if "starts before" in hypothesis or "starts after" in hypothesis:
            total_start += 1
            if answer_label == prediction_label:
                correct_start += 1
        else:
            total_end += 1
            if answer_label == prediction_label:
                correct_end += 1
    s_total = 0
    s_correct = 0
    for key in story_prediction_map:
        s_total += 1
        cv = True
        for v in story_prediction_map[key]:
            cv = cv and v
        if cv:
            s_correct += 1
    total_acc = float(correct) / float(total)
    start_acc = float(correct_start) / float(total_start)
    end_acc = float(correct_end) / float(total_end)
    story_em = float(s_correct) / float(s_total)
    return total_acc, start_acc, end_acc, story_em


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate leaderboard predictions for questions."
    )

    parser.add_argument(
        "--question_answers",
        "-qa",
        help="Filename of the question answers to read.",
        required=True,
    )
    parser.add_argument(
        "--predictions",
        "-p",
        help="Filename of the leaderboard predictions",
        required=True,
    )
    parser.add_argument(
        "--output", "-o", help="Output results to this file.", required=True
    )

    args = parser.parse_args()


    total_acc, start_acc, end_acc, story_em = evaluate(
        args.question_answers, args.predictions
    )

    with open(args.output, "wt", encoding="UTF-8") as output:
        output.write(
            json.dumps(
                {
                    "total_acc": total_acc,
                    "start_acc": start_acc,
                    "end_acc": end_acc,
                    "story_em": story_em,
                }
            )
        )


if __name__ == "__main__":
    main()
