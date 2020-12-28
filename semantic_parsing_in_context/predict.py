from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
# WARNING: Do not exclude these imports
from predictor.sparc_predictor import SparcPredictor
from dataset_reader.sparc_reader import SparcDatasetReader
from models.sparc_parser import SparcParser


class PredictManager:

    def __init__(self, archive_file, tables_file, database_path):
        overrides = "{\"dataset_reader.tables_file\":\"" + tables_file + "\",\"dataset_reader.database_path\":" +\
                    "\"" + database_path + "\"}"
        archive = load_archive(archive_file,
                               overrides=overrides)
        self.predictor = Predictor.from_archive(
            archive, predictor_name="sparc")

    def predict_result(self, ques_inter: str, ques_database: str):
        param = {
            "database_id": ques_database,
            "question": ques_inter
        }
        restate = self.predictor.predict_json(param)["best_predict_sql"]
        return restate


if __name__ == '__main__':
    manager = PredictManager(archive_file="model.tar.gz",
                             tables_file="dataset_sparc/tables.json",
                             database_path="dataset_sparc/database")
    # the input dialogue is separate by `;`, and the second argument is database_id
    result = manager.predict_result("What are all the airlines;Of these, which is Jetblue Airways", "flight_2")
    print(result)
