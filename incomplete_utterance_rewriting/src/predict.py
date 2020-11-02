from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
# WARNING: Do not exclude these imports
from predictor import RewritePredictor
from data_reader import RewriteDatasetReader
from model import UnifiedFollowUp


class PredictManager:

    def __init__(self, archive_file):
        archive = load_archive(archive_file)
        self.predictor = Predictor.from_archive(
            archive, predictor_name="rewrite")

    def predict_result(self, dialog_flatten: str):
        # dialog_flatten is split by \t\t
        dialog_snippets = dialog_flatten.split("\t\t")
        param = {
            "context": dialog_snippets[:-1],
            "current": dialog_snippets[-1]
        }
        restate = self.predictor.predict_json(param)["predicted_tokens"]
        return restate


if __name__ == '__main__':
    manager = PredictManager("../pretrained_weights/multi_bert.tar.gz")
    result = manager.predict_result("周 末 就 要 上 班 了		我 也 是		节 操 在 哪 里		年 终 奖 多 少		你 呢")
    print(result)
