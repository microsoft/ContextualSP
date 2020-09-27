# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register("sparc")
class SparcPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        # Now get result
        results = self.predict_instance(instance)
        return results

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "table": "..."}``.
        """
        utter_list = []
        if "interaction" in json_dict:
            """
            predict mode
            """
            for ins in json_dict["interaction"]:
                utter_list.append(ins["utterance"])
        else:
            """
            demo mode
            """
            utter_list = json_dict["question"].split(";")

        db_id = json_dict["database_id"]

        instance = self._dataset_reader.text_to_instance(utter_list,  # type: ignore
                                                         db_id)
        return instance
