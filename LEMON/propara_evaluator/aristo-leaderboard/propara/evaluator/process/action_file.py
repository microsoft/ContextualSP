from collections import OrderedDict, defaultdict
from typing import NamedTuple, Dict, List

from errors import corrupted_action_file
from process.constants import LOCATION_UNKNOWN, NO_LOCATION, NO_ACTION, CREATE, MOVE, DESTROY
from process import ProcessSummary, Process


def _accumulate_action(locations, actions, num_steps, participant, action, before_location, after_location, step_id):
    existing_locations = locations.setdefault(participant, [LOCATION_UNKNOWN] * (1 + num_steps))
    existing_actions = actions.setdefault(participant, [NO_ACTION] * num_steps)

    if step_id == 1:
        existing_locations[0] = before_location

    existing_locations[step_id] = after_location
    existing_actions[step_id - 1] = action

    return locations, actions


def _num_sentences_in_actions_file(actions_filename: str) -> Dict[int, int]:
    num_sentences = defaultdict(int)  # type: Dict[int, int]
    with open(actions_filename) as f:
        line_num = 0
        for line in f:
            line_num += 1
            try:
                process_id_str, step_id_str = line.strip().split('\t', 2)[:2]
            except ValueError as e:
                corrupted_action_file(
                    filename=actions_filename,
                    line_num=line_num,
                    details=str(e)
                )

            process_id = int(process_id_str)
            step_id = int(step_id_str)

            num_sentences[process_id] = max(num_sentences[process_id], step_id)

    if not num_sentences:
        corrupted_action_file(actions_filename, "no lines to iterate")

    return num_sentences


class ActionFile(NamedTuple):
    filename: str

    # key = process_id
    # value = OrderedDict like this:
    #   key = participant string (like "water vapor ; lifted vapor ; vapor")
    #   value = list of location strings, length = 1 + number of sentences
    locations: Dict[int, Dict[str, List[str]]]

    # key = process_id
    # value = OrderedDict like this:
    #   key = participant string (like "water vapor ; lifted vapor ; vapor")
    #   value = list of actions (CREATE, DESTROY, MOVE or NONE), length = number of sentences
    actions: Dict[int, Dict[str, List[str]]]

    # key = process_id
    # value = number of sentences per process
    num_sentences: Dict[int, int]

    def has_process_id(self, process_id: int):
        return process_id in self.locations

    def summarize(self) -> Dict[int, ProcessSummary]:
        summary_by_process_id = dict()  # type: Dict[int, ProcessSummary]
        for process_id in self.locations.keys():
            locations = self.locations[process_id]
            actions = self.actions[process_id]

            p = Process(process_id=process_id, locations=locations, actions=actions,
                        num_steps=self.num_sentences[process_id])

            summary_by_process_id[p.process_id] = ProcessSummary(
                process_id=p.process_id,
                inputs=p.inputs(),
                outputs=p.outputs(),
                conversions=p.conversions(),
                moves=p.moves(),
            )

        return summary_by_process_id

    def diff_participants(self, other: "ActionFile") -> List[str]:
        report: List[str] = []

        for process_id in self.process_ids():
            self_participants = self.participants(process_id)

            if not other.has_process_id(process_id):
                report.append(f"Process {process_id} missing in {other.filename}")
                continue

            other_participants = other.participants(process_id)

            process_report: List[str] = []
            for p in self_participants:
                if p not in other_participants:
                    process_report.append(f"Process {process_id} in {other.filename}: participant \"{p}\" is missing.")

            for op in other_participants:
                if op not in self_participants:
                    process_report.append(
                        f"Process {process_id} in {other.filename}: participant \"{op}\" is unexpected.")

            report += sorted(process_report)

        return report

    def process_ids(self) -> List[int]:
        return sorted(self.locations.keys())

    def participants(self, process_id) -> List[str]:
        return sorted(self.locations[process_id].keys())

    # Reads an actionfile from disk.
    @classmethod
    def from_file(cls, action_filename: str) -> "ActionFile":
        num_sentences = _num_sentences_in_actions_file(action_filename)
        locations = defaultdict(OrderedDict)  # type: Dict[int, Dict[str, List[str]]]
        actions = defaultdict(OrderedDict)  # type: Dict[int, Dict[str, List[str]]]

        line_num = 0
        with open(action_filename) as f:
            for line in f:
                line_num += 1
                try:
                    process_id_str, step_id_str, participant, action, before_location, after_location = \
                        line.strip("\n\r").split('\t', 6)[:6]
                except ValueError as e:
                    corrupted_action_file(
                        filename=action_filename,
                        line_num=line_num,
                        details=str(e)
                    )

                process_id = int(process_id_str)
                step_id = int(step_id_str)

                if action == NO_ACTION:
                    if before_location != after_location:
                        corrupted_action_file(
                            filename=action_filename,
                            line_num=line_num,
                            details=f"Unequal NONE locations: {before_location} -- {after_location}"
                        )
                elif action == CREATE:
                    if before_location != '-':
                        corrupted_action_file(
                            filename=action_filename,
                            line_num=line_num,
                            details=f"Invalid CREATE before_location: {before_location}"
                        )
                    before_location = NO_LOCATION
                    if after_location == "" or after_location == '-':
                        corrupted_action_file(
                            filename=action_filename,
                            line_num=line_num,
                            details=f"Invalid CREATE after_location: {after_location}"
                        )
                elif action == DESTROY:
                    if before_location == "" or before_location == '-':
                        corrupted_action_file(
                            filename=action_filename,
                            line_num=line_num,
                            details=f"Invalid DESTROY before_location: {before_location}"
                        )
                    if after_location != '-':
                        corrupted_action_file(
                            filename=action_filename,
                            line_num=line_num,
                            details=f"Invalid DESTROY after_location: {after_location}"
                        )
                elif action == MOVE:
                    if before_location == "" or before_location == '-':
                        corrupted_action_file(
                            filename=action_filename,
                            line_num=line_num,
                            details=f"Invalid MOVE before_location: {before_location}"
                        )
                    if after_location == "" or after_location == '-':
                        corrupted_action_file(
                            filename=action_filename,
                            line_num=line_num,
                            details=f"Invalid MOVE after_location: {after_location}"
                        )
                else:
                    corrupted_action_file(
                        filename=action_filename,
                        line_num=line_num,
                        details=f"Invalid action: {action}"
                    )

                if before_location == "-":
                    before_location = NO_LOCATION
                elif before_location == "?":
                    before_location = LOCATION_UNKNOWN

                if after_location == "-":
                    after_location = NO_LOCATION
                elif after_location == "?":
                    after_location = LOCATION_UNKNOWN

                # update locations and actions for this process_id
                locations[process_id], actions[process_id] = \
                    _accumulate_action(
                        locations[process_id],
                        actions[process_id],
                        num_sentences[process_id],
                        participant,
                        action,
                        before_location,
                        after_location,
                        step_id,
                    )

        if not locations:
            corrupted_action_file(action_filename, "no lines to iterate")

        return cls(
            filename=action_filename,
            locations=locations,
            actions=actions,
            num_sentences=num_sentences
        )
