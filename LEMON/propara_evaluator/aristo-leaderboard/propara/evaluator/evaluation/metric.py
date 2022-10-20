from typing import Dict, NamedTuple


class Metric(NamedTuple):
    precision: float
    recall: float

    def F1(self):
        if self.precision + self.recall == 0:
            return 0.0

        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def diagnostics(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall
        }
