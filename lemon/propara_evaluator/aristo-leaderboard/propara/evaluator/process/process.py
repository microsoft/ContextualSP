from typing import List, NamedTuple, Dict

from process.constants import NO_LOCATION, CREATE, DESTROY, MOVE


class Input(NamedTuple):
    participants: str


class Output(NamedTuple):
    participants: str


class Conversion(NamedTuple):
    created: str
    destroyed: str
    locations: str
    step_id: str


class Move(NamedTuple):
    participants: str
    location_before: str
    location_after: str
    step_id: str


class Process(NamedTuple):
    process_id: int
    locations: Dict
    actions: Dict
    num_steps: int

    #  Q1: What are the inputs?
    #      - If a participant exists in state1, but does not exist in the end stateN, it's an input.
    def inputs(self) -> List[Input]:
        inputs = []  # type: List[Input]
        for participant in self.locations.keys():
            actions = self.actions[participant]

            if _is_this_action_seq_of_an_input(actions):
                inputs.append(Input(participants=_summarize_participants(participant)))
        return inputs

    #  Q2: What are the outputs
    #      - If a participant does not exist in state1, but exists in the end stateN, it's an output.
    def outputs(self) -> List[Output]:
        outputs = []  # type: List[Output]
        for participant in self.locations.keys():
            actions = self.actions[participant]

            if _is_this_action_seq_of_an_output(actions):
                outputs.append(Output(participants=_summarize_participants(participant)))
        return outputs

    #  Q3: What is converted?
    #      tuple: (participant-list-from, participant-list-to, loc-list, step-id)
    #      a. For any event with BOTH "D" and "C" in:
    #       	The "D" participants are converted to the "C" participants at the union of the D and C locations
    #      b. IF an event has ONLY "D" but no "C" in   ("M" is ok - irrelevant)
    #       	AND the NEXT event has ONLY "C" but no "D" in   ("M" is ok - irrelevant)
    #       	THEN the "D" participants are converted to the "C" participants at the union of the D and C locations
    def conversions(self) -> List[Conversion]:
        conversions = []  # type: List[Conversion]
        for step_id in range(1, self.num_steps + 1):
            (created, c_locations) = self._get_created_at_step(step_id)
            (destroyed, d_locations) = self._get_destroyed_at_step(step_id)
            if created and destroyed:
                conversions.append(Conversion(
                    destroyed=_conjunction(*destroyed),
                    created=_conjunction(*created),
                    locations=_conjunction(*set(c_locations + d_locations)),
                    step_id=str(step_id)
                ))
            elif destroyed and step_id < self.num_steps - 1:
                (created2, c_locations2) = self._get_created_at_step(step_id + 1)
                (destroyed2, d_locations2) = self._get_destroyed_at_step(step_id + 1)
                created_but_not_destroyed = set(created2) - set(destroyed)
                if not destroyed2 and created_but_not_destroyed:
                    conversions.append(Conversion(
                        destroyed=_conjunction(*destroyed),
                        created=_conjunction(*created_but_not_destroyed),
                        locations=_conjunction(*set(c_locations2 + d_locations)),
                        step_id=str(step_id)
                    ))
            elif created and step_id < self.num_steps - 1:
                (created2, c_locations2) = self._get_created_at_step(step_id + 1)
                (destroyed2, d_locations2) = self._get_destroyed_at_step(step_id + 1)
                destroyed_but_not_created = set(destroyed2) - set(created)
                if not created2 and destroyed_but_not_created:
                    conversions.append(Conversion(
                        destroyed=_conjunction(*destroyed_but_not_created),
                        created=_conjunction(*created),
                        locations=_conjunction(*set(c_locations + d_locations2)),
                        step_id=str(step_id)
                    ))

        return conversions

    #  Q4: What is moved?
    #      tuple: (participant, from-loc, to-loc, step-id)
    #  return all moves
    def moves(self):
        moves = []
        for participant in self.locations.keys():
            locations = self.locations[participant]
            actions = self.actions[participant]

            for step_id in range(1, len(locations)):
                is_moved = actions[step_id - 1] == MOVE or (
                        locations[step_id - 1] != NO_LOCATION and
                        locations[step_id] != NO_LOCATION and
                        locations[step_id - 1] != locations[step_id]
                )

                if not is_moved:
                    continue

                moves.append(Move(
                    participants=_summarize_participants(participant),
                    location_before=locations[step_id - 1],
                    location_after=locations[step_id],
                    step_id=str(step_id)
                ))

        return moves

    def _get_created_at_step(self, step_id: int):
        created = []
        locations = []

        for participant in self.locations.keys():
            state_values = self.locations[participant]
            is_creation = state_values[step_id - 1] == NO_LOCATION \
                          and state_values[step_id] != NO_LOCATION
            if is_creation:
                created.append(_summarize_participants(participant))
                locations.append(state_values[step_id])

        return created, locations

    def _get_destroyed_at_step(self, step_id: int):
        destroyed = []
        locations = []

        for participant in self.locations.keys():
            state_values = self.locations[participant]
            is_destruction = state_values[step_id - 1] != NO_LOCATION \
                             and state_values[step_id] == NO_LOCATION
            if is_destruction:
                destroyed.append(_summarize_participants(participant))
                locations.append(state_values[step_id - 1])

        return destroyed, locations


def _is_this_action_seq_of_an_output(actions) -> bool:
    for action_id, _ in enumerate(actions):
        no_destroy_move_before = DESTROY not in actions[0:action_id] and MOVE not in actions[0:action_id]
        current_create = actions[action_id] == CREATE
        no_destroy_later = DESTROY not in actions[action_id + 1:]
        if no_destroy_move_before and current_create and no_destroy_later:
            return True
    return False


def _is_this_action_seq_of_an_input(actions) -> bool:
    for action_id, _ in enumerate(actions):
        no_create_before = CREATE not in actions[0:action_id]  # last action_id must be checked
        current_destroy = actions[action_id] == DESTROY
        no_create_move_later = CREATE not in actions[action_id + 1:] and MOVE not in actions[action_id + 1:]

        if no_create_before and current_destroy and no_create_move_later:
            return True
    return False


def _split_participants(participant) -> List[str]:
    return [p.strip() for p in participant.split(';')]


def _summarize_participants(participant) -> str:
    return ' OR '.join(_split_participants(participant))


def _conjunction(*things) -> str:
    return ' AND '.join(things)
