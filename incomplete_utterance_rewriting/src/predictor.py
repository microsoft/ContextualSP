from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("rewrite")
class RewritePredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        context = json_dict["context"]
        current = json_dict["current"]
        # placeholder
        restate = "hi"

        return self._dataset_reader.text_to_instance(context, current, restate, training=False)
