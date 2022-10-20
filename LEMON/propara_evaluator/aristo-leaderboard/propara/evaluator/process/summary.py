from typing import Dict, List, NamedTuple

from process.process import Conversion, Move, Input, Output


class ProcessSummary(NamedTuple):
    process_id: int
    inputs: List[Input]
    outputs: List[Output]
    conversions: List[Conversion]
    moves: List[Move]

    def __repr__(self):
        return f"Process {self.process_id}" \
               f" inputs({self.inputs})" \
               f" outputs({self.outputs})" \
               f" conversions({self.conversions})" \
               f" moves({self.moves})"

    def diagnostics(self) -> Dict:
        return {
            "process_id": self.process_id,
            "inputs": self._inputs_diagnostics(),
            "outputs": self._outputs_diagnostics(),
            "conversions": self._conversions_diagnostics(),
            "moves": self._moves_diagnostics(),
        }

    def _inputs_diagnostics(self):
        inputs = []
        for i in self.inputs:
            inputs.append(i.participants)
        if len(inputs) > 0:
            return {"participants": inputs}
        return {"participants": None}

    def _outputs_diagnostics(self):
        outputs = []
        for i in self.outputs:
            outputs.append(i.participants)
        if len(outputs) > 0:
            return {"participants": outputs}
        return {"participants": None}

    def _conversions_diagnostics(self):
        conversions = []
        for c in self.conversions:
            conversions.append({
                "participants_destroyed": c.destroyed,
                "participants_created": c.created,
                "location": c.locations,
                "step_number": int(c.step_id),
            })

        if len(conversions) > 0:
            return conversions
        return None

    def _moves_diagnostics(self):
        moves = []
        for m in self.moves:
            moves.append({
                "participants": m.participants,
                "location_before": m.location_before,
                "location_after": m.location_after,
                "step_number": int(m.step_id),
            })

        if len(moves) > 0:
            return moves
        return None
