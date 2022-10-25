from codecs import open

from strongsup.example import Context, Example
from strongsup.example_factory import ExampleFactory
from strongsup.rlong.state import RLongAlchemyState, RLongSceneState, RLongTangramsState, RLongUndogramsState
from strongsup.rlong.value import RLongStateValue
from strongsup.rlong.world import RLongAlchemyWorld, RLongSceneWorld, RLongTangramsWorld, RLongUndogramsWorld


################################
# RLongExampleFactory

class RLongExampleFactory(ExampleFactory):

    def __init__(self, filename, domain_name, num_steps_list,
            slice_steps_from_middle):
        """Read RLongDataset.

        Args:
            domain_name (str): 'alchemy', 'scene', 'tangrams', or 'undograms'
            filename (str): TSV File to load data from. The file format is
                <id> <initstate> <sentence1> <state1> <sentence2> <state2> ...
            num_steps_list (list[int]): Number of sentences for each example.
                E.g., [2, 3] creates examples from the first 2 or 3 sentences.
                num_steps of -1 will take all utterances.
            slice_steps_from_middle (bool): Whether to also get the sentences
                from the middle of the stories. Setting this to False will only
                get the sentences from the beginning of the stories.
        """
        self._filename = filename
        self._domain_name = domain_name
        if domain_name == 'alchemy':
            self._state_class = RLongAlchemyState
            self._world_class = RLongAlchemyWorld
        elif domain_name == 'scene':
            self._state_class = RLongSceneState
            self._world_class = RLongSceneWorld
        elif domain_name == 'tangrams':
            self._state_class = RLongTangramsState
            self._world_class = RLongTangramsWorld
        elif domain_name == 'undograms':
            self._state_class = RLongUndogramsState
            self._world_class = RLongUndogramsWorld
        else:
            raise ValueError('Unknown rlong domain name: {}'.format(domain_name))

        # Parse num_steps
        if not isinstance(num_steps_list, list):
            assert isinstance(num_steps_list, int)
            num_steps_list = list([num_steps_list])
        self._num_steps_list = num_steps_list

        self._slice_steps_from_middle = slice_steps_from_middle

    @property
    def examples(self):
        with open(self._filename, 'r', 'utf8') as fin:
            for line in fin:
                line = line.rstrip('\n').split('\t')
                assert len(line) % 2 == 0
                for num_steps in self._num_steps_list:
                    if num_steps == -1:
                        # Maximum number of steps
                        num_steps = len(line) / 2 - 1
                    start_idx = 1
                    while start_idx + 2 * num_steps < len(line):
                        utterances = [utterance.split() for utterance in
                                line[start_idx+1:start_idx+2*num_steps:2]]
                        init_state = self._state_class.from_raw_string(
                                line[start_idx])
                        target_state = self._state_class.from_raw_string(
                                line[start_idx+2*num_steps])
                        world = self._world_class(init_state)
                        context = Context(world, utterances)
                        example = Example(context,
                                answer=[RLongStateValue(target_state)])
                        yield example
                        if not self._slice_steps_from_middle:
                            break
                        start_idx += 2
