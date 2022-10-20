import json
import random
import sys

from allennlp_reasoning_explainqa.common.constants import CORRECT_OPTION_TAG
from allennlp_reasoning_explainqa.training.metrics.confusion_matrix import (
    F1MeasureCustomRetrievalEval,
)
from allennlp_reasoning_explainqa.training.metrics.explanation_eval import (
    ExplanationEval,
)

# Sets random seed to a nothing-up-my-sleeve number so that we have
# deterministic evaluation scores.
random.seed(12345)

# Sets random seed to a nothing-up-my-sleeve number so that we have
# deterministic evaluation scores.
random.seed(12345)


def evaluate(prediction_filename, label_filename):
    chainid_to_label = json.load(open(label_filename, "r"))
    chain_count = len(chainid_to_label)

    predictions_lines = open(prediction_filename, "r").readlines()
    predictions = [json.loads(row) for row in predictions_lines]
    prediction_count = len(predictions)
    if chain_count != prediction_count:
        print(
            f"Label file {label_filename} has {chain_count} chains, but prediction file {prediction_filename} has {prediction_count} predictions. These must be equal."
        )
        sys.exit(1)

    f1eval = F1MeasureCustomRetrievalEval(pos_label=1)
    explanation_eval = ExplanationEval()
    chain_ids_covered = []

    cnt = 0
    for row in predictions:
        assert "score" in row, "Prediction should contain field score"
        assert "chain_id" in row, "Prediction should contain field chain_id"
        score = row["score"]
        chain_id = row["chain_id"]
        qid = chain_id.strip().split("_")[0]
        print("qid,chain_id,score = ", qid, chain_id, score)
        gtlabel = chainid_to_label[chain_id]
        f1eval(int(gtlabel), score)
        explanation_eval(qid, CORRECT_OPTION_TAG, int(gtlabel), score)
        chain_ids_covered.append(chain_id)
        cnt += 1

    assert len(chain_ids_covered) == len(
        chainid_to_label
    ), "Found {} chains but expected {} chains".format(
        len(chain_ids_covered), len(chainid_to_label)
    )
    binclf_performance = f1eval.get_metric(reset=True)
    print("f1.get_metric() = ", binclf_performance)
    explanation_performance = explanation_eval.get_metric(reset=True)
    print("explanation_eval.get_metric() = ", explanation_performance)
    final_metrics = {
        "auc_roc": binclf_performance["auc_roc"],
        "explainP1": explanation_performance["explainP1"],
        "explainNDCG": explanation_performance["explainNDCG"],
    }
    print("=" * 32)
    print(": auc_roc = ", binclf_performance["auc_roc"])
    print(": P1 = ", explanation_performance["explainP1"])
    print(": explainNDCG = ", explanation_performance["explainNDCG"])
    print("=" * 32)
    return final_metrics


if __name__ == "__main__":
    prediction_filename = sys.argv[1]
    label_filename = sys.argv[2]
    metrics_filename = sys.argv[3]

    print(
        f"Evaluating prediction file {prediction_filename} with label file {label_filename}"
    )
    metrics = evaluate(prediction_filename, label_filename)

    print(f"Writing final metrics to file: {metrics_filename}")
    json.dump(metrics, open(metrics_filename, "w"))
