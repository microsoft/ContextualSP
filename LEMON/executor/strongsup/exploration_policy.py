import numpy as np

from collections import Sequence, Counter

from abc import ABCMeta, abstractmethod

from gtd.chrono import verboserate
from gtd.utils import flatten
from strongsup.parse_case import ParseCase, ParsePath
from strongsup.utils import epsilon_greedy_sample, softmax
from strongsup.utils import sample_with_replacement
from strongsup.decoder import NormalizationOptions


class Beam(Sequence):
    """A Sequence of ParsePaths.
    In each ParsePath, each ParseCase must have already have a decision.

    Usually paths in a Beam are unique, but this is not required
    (e.g., BatchedReinforce uses Beams with repeated paths).
    """
    __slots__ = ['_paths']

    @classmethod
    def initial_beam(self, context):
        """Return the initial beam for the context.

        Args:
            context (Context)

        Returns:
            Beam
        """
        return Beam([ParsePath.empty(context)])

    def __init__(self, paths):
        self._paths = paths

    def __getitem__(self, i):
        return self._paths[i]

    def __len__(self):
        return len(self._paths)

    def __str__(self):
        return 'Beam' + str(self._paths)
    __repr__ = __str__

    def append(self, path):
        self._paths.append(path)

    @property
    def terminated(self):
        """Whether all paths are terminated."""
        return all(path.terminated for path in self._paths)

    def get_terminated(self):
        """Get only the terminated paths."""
        return Beam([path for path in self._paths if path.terminated])


def get_num_iterations(iterations_per_utterance, examples):
    """Returns the number of iterations to run for in this batch of examples

    Args:
        iterations_per_utterance (int): iterations per utterance config
        examples (list[Example])

    Returns:
        int: number of iterations
    """
    return iterations_per_utterance * max(
            [len(ex.context.utterances) for ex in examples])


class ExplorationPolicy(object, metaclass=ABCMeta):
    """For given examples, search for candidate ParseCase based on some
    exploration policy.

    An ExplorationPolicy will be called by the decoder.
    Since Examples are passed in, an ExplorationPolicy can choose to 'cheat'
    and use the answer or gold logical form to aid the exploration.
    This is totally fine during training.
    """

    def __init__(self, decoder, config, normalization, train):
        """
        Args:
            decoder (Decoder)
            config (Config)
            normalization (NormalizationOptions)
            train (bool): train or test policy
        """
        self._decoder = decoder
        self._config = config
        self._normalization = normalization
        self._train = train

    @abstractmethod
    def get_beams(self, examples, verbose=False):
        """Return a beam of scored ParseCases for each example.

        Args:
            examples (list[Example]): List of examples
            verbose (bool): Verbosity
        Returns:
            list[Beam] of length len(examples).
        """
        raise NotImplementedError

    @abstractmethod
    def get_intermediate_beams(self, examples, verbose=False):
        """Return the final beam along with intermediate beams / exploration states.

        Args:
            examples (list[Example]): List of examples
            verbose (bool): Verbosity
        Returns:
            list[Beam], list[list[Beam]]
            Each list has length len(examples).
            Each sublist i in the second output contains the intermediate beams
                for example i.
        """
        raise NotImplementedError

    def _ranker(self, path):
        """Assigns a score to a ParsePath depending on the configs

        Return the log unnormalized probability of the ParsePath.
        The returned value can be used to rank ParsePaths.

        For local normalization, the method returns the log-probability.
        For global normalization, the method returns the cumulative logit.

        Args:
            path (ParsePath): path to be scored

        Return:
            float: the score
        """
        if self._normalization == NormalizationOptions.LOCAL:
            return path.log_prob
        elif self._normalization == NormalizationOptions.GLOBAL:
            return path.score
        else:
            raise ValueError(
                'Unknown normalization type: {}'.format(self._normalization))


################################
# Beam search

class BeamSearchExplorationPolicy(ExplorationPolicy):
    def __init__(self, decoder, config, normalization, train):
        super(BeamSearchExplorationPolicy, self).__init__(
                decoder, config, normalization, train)
        if not train:
            assert not config.independent_utterance_exploration
            assert config.exploration_epsilon == 0

    def get_beams(self, examples, verbose=False):
        beams = [Beam.initial_beam(ex.context) for ex in examples]
        num_iterations = get_num_iterations(
                self._config.iterations_per_utterance, examples)
        if verbose:
            iterations = verboserate(list(range(num_iterations)),
                                     desc='Performing beam search')
        else:
            iterations = range(num_iterations)
        for _ in iterations:
            beams = self.advance(beams)
        return beams

    def get_intermediate_beams(self, examples, verbose=False):
        beams = [Beam.initial_beam(ex.context) for ex in examples]
        intermediates = [[] for _ in examples]
        num_iterations = get_num_iterations(
                self._config.iterations_per_utterance, examples)
        if verbose:
            iterations = verboserate(list(range(num_iterations)),
                                     desc='Performing beam search')
        else:
            iterations = range(num_iterations)
        for _ in iterations:
            for ex_idx, beam in enumerate(beams):
                intermediates[ex_idx].append(beam)
            beams = self.advance(beams)
        return beams, intermediates

    def advance(self, beams):
        """Advance a batch of beams.

        Args:
            beams (list[Beam]): a batch of beams

        Returns:
            list[Beam]: a new batch of beams
                (in the same order as the input beams)
        """
        # Gather everything needed to be scored
        # For efficiency, pad so that the number of cases from each beam
        #   is equal to beam_size.
        cases_to_be_scored = []
        new_paths = []
        for beam in beams:
            # terminated stores terminated paths
            # which do not count toward the beam size limit
            terminated = []
            # unterminated stores unterminated paths and a partial ParseCase
            # containing the possible candidate choices
            unterminated = []
            num_cases_to_be_scored = 0
            #print '@' * 40
            for path in beam:
                if path.terminated:
                    terminated.append(path)
                else:
                    case = path.extend()
                    unterminated.append((path, case))
                    cases_to_be_scored.append(case)
                    num_cases_to_be_scored += 1
            new_paths.append((terminated, unterminated))
            # Pad to beam_size
            assert num_cases_to_be_scored <= self._config.beam_size

            if beam:
                while num_cases_to_be_scored < self._config.beam_size:
                    case = ParseCase.initial(beam[0].context)
                    cases_to_be_scored.append(case)
                    num_cases_to_be_scored += 1

        # for exploration, use a parser which pretends like every utterance
        # is the first utterance it is seeing
        ignore_previous_utterances = \
                self._config.independent_utterance_exploration

        # Use the ParseModel to score
        self._decoder.parse_model.score(cases_to_be_scored,
                                        ignore_previous_utterances,
                                        self._decoder.caching)

        # Read the scores and create new paths
        new_beams = []
        #print '=' * 40
        for terminated, unterminated in new_paths:
            #print '-' * 20
            new_unterminated = []
            for path, case in unterminated:
                for choice in case.choices:
                    clone = case.copy_with_decision(choice)
                    denotation = clone.denotation
                    # Filter out the cases with invalid denotation
                    if not isinstance(denotation, Exception):
                        path = clone.path
                        if path.terminated:
                            try:
                                # Test if the denotation can be finalized
                                path.finalized_denotation
                                #print 'FOUND [T]', clone.path.decisions, denotation, denotation.utterance_idx, path.finalized_denotation
                                terminated.append(path)
                            except ValueError as e:
                                #print 'FOUND [BAD T]', e
                                pass
                        elif self._decoder.path_checker(path):
                            #print 'FOUND', clone.path.decisions, denotation, denotation.utterance_idx
                            new_unterminated.append(path)
                        else:
                            #print 'PRUNED', clone.path.decisions, denotation, denotation.utterance_idx
                            pass
                    else:
                        #print 'BAD', clone.path.decisions, denotation
                        pass
            # Sort the paths
            terminated.sort(key=self._ranker, reverse=True)
            new_unterminated.sort(key=self._ranker, reverse=True)
            # Prune to beam size with exploration
            epsilon = self._config.exploration_epsilon
            selected = epsilon_greedy_sample(
                    new_unterminated,
                    min(self._config.beam_size, len(new_unterminated)),
                    epsilon=epsilon)
            # Create a beam from the remaining paths
            new_beams.append(Beam(terminated + selected))
        return new_beams


################################
# Stale Beam Search

class BeamMetaInfo(object):
    """Wrapper around a Beam that includes metadata for BeamMap"""
    def __init__(self, beam, age):
        self._beam = beam
        self._age = age

    @property
    def beam(self):
        return self._beam

    @property
    def age(self):
        return self._age

    def increment_age(self):
        self._age += 1


class BeamMap(object):
    """Maintains a map between Example and stale Beams"""

    def __init__(self):
        self._map = {}  # example --> BeamMetaInfo

    def contains(self, example):
        """Returns if example is in the map

        Args:
            example (Example)

        Returns:
            bool: True if example in map
        """
        return example in self._map

    def get_beam_age(self, example):
        """Returns how old the beam for this example is

        Args:
            example (Example)

        Returns:
            int: the age
        """
        assert self.contains(example)

        return self._map[example].age

    def increment_age(self, example):
        """Increments the age of the beam associated with this example

        Args:
            example (Example)
        """
        assert example in self._map
        self._map[example].increment_age()

    def set_beam(self, example, beam):
        """Sets the beam associated with this example.

        Args:
            example (Example)
            beam (Beam)
        """
        self._map[example] = BeamMetaInfo(beam, 1)

    def get_beam(self, example):
        """Returns the beam associated with this example.

        Args:
            example (Example)

        Returns:
            Beam
        """
        assert example in self._map
        return self._map[example].beam


class StaleBeamSearch(ExplorationPolicy):
    """Performs beam search every max_age iterations.
    On the other iterations, returns the stale beams.
    NOTE: Does not recalculate scores

    Args:
        decoder (Decoder)
        config (Config)
        normalization (NormalizationOptions)
        fresh_policy (ExplorationPolicy): the policy that runs to obtain
            fresh beams
        train (bool): train or test policy
    """
    def __init__(self, decoder, config, normalization, train):
        if not train:
            raise ValueError(
                    "Stale Beam Search should only be used at train time")
        super(StaleBeamSearch, self).__init__(
                decoder, config, normalization, train)
        self._fresh_policy = get_exploration_policy(
                decoder, config.fresh_policy, normalization, train)
        self._max_age = self._config.max_age  # iterations till refresh
        self._beam_map = BeamMap()

    def get_beams(self, examples, verbose=False):
        expired_examples = []  # Needs to be updated with BeamSearch
        fresh_beams = []  # Fetched from cache
        fresh_indices = []  # True @i if example i is fresh
        for example in examples:
            # Non-existent or expired
            if not self._beam_map.contains(example) or \
                    self._beam_map.get_beam_age(example) >= self._max_age:
                fresh_indices.append(False)
                expired_examples.append(example)
            else:  # Still fresh
                self._beam_map.increment_age(example)
                fresh_indices.append(True)
                fresh_beams.append(self._beam_map.get_beam(example))

        # Recalculate expired beams
        if len(expired_examples) > 0:
            recalculated_beams = self._fresh_policy.get_beams(
                    expired_examples, verbose)
        else:
            recalculated_beams = []

        # Cache recalculated beams
        for expired_example, recalculated_beam in zip(
                expired_examples, recalculated_beams):
            self._beam_map.set_beam(expired_example, recalculated_beam)

        # Put beams back in correct order
        beams = []
        for fresh in fresh_indices:
            if fresh:
                beams.append(fresh_beams.pop(0))
            else:
                beams.append(recalculated_beams.pop(0))

        return beams

    def get_intermediate_beams(self, examples, verbose=False):
        return self._fresh_policy.get_intermediate_beams(
                examples, verbose=verbose)


################################
# Gamma Sampling ABC

class GammaSamplingExplorationPolicy(ExplorationPolicy, metaclass=ABCMeta):
    """Creates a beam using some form of sampling."""

    def __init__(self, decoder, config, normalization, train):
        if not train:
            raise ValueError(
                    "Sampling Exploration should only be used at train time.")
        super(GammaSamplingExplorationPolicy, self).__init__(
                decoder, config, normalization, train)
        assert config.exploration_epsilon is None

    def get_beams(self, examples, verbose=False):
        terminated = [set() for _ in examples]
        # initialize beams
        beams = [[ParsePath.empty(ex.context)] for ex in examples]
        # put all probability mass on the root
        distributions = [[1] for _ in examples]
        num_iterations = get_num_iterations(
                self._config.iterations_per_utterance, examples)

        iterations = range(num_iterations)
        if verbose:
            iterations = verboserate(
                    iterations, desc='Performing randomized search')

        for _ in iterations:
            terminated, beams, distributions = self.advance(
                    terminated, beams, distributions)

        return [Beam(sorted(list(paths), key=self._ranker, reverse=True))
                for paths in terminated]

    def get_intermediate_beams(self, examples, verbose=False):
        intermediates = [[] for _ in examples]

        terminated = [set() for ex in examples]
        particles = [[ParsePath.empty(ex.context)] for ex in examples]
        distributions = [[1] for _ in range(len(examples))]
        num_iterations = get_num_iterations(
                self._config.iterations_per_utterance, examples)

        if verbose:
            iterations = verboserate(list(range(num_iterations)),
                                     desc='Performing randomized search')
        else:
            iterations = range(num_iterations)

        for _ in iterations:
            for ex_idx, (beam, terminated_set) in enumerate(
                    zip(particles, terminated)):
                intermediates[ex_idx].append(Beam(
                    sorted(terminated_set, key=self._ranker, reverse=True) +
                    sorted(beam, key=self._ranker, reverse=True)))
            terminated, particles, distributions = self.advance(
                    terminated, particles, distributions)

        return [Beam(sorted(list(paths), key=self._ranker, reverse=True))
                for paths in terminated], intermediates

    def advance(self, terminated, beams, empirical_distributions):
        """Advance a batch of beams.

        Args:
            terminated (list[set(ParsePath)]): a batch of all the
                terminated paths found so far for each beam.
            beams (list[list[ParsePath]]): a batch of beams.
                All paths on all beams have the same length (all
                should be unterminated)
            empirical_distributions (list[list[float]]): a batch of
                distributions over the corresponding beams.

        Returns:
            list[set[ParsePath]]: a batch of terminated beams
                (in the same order as the input beams)
            list[list[ParsePath]]: a batch of new beams all extended
                by one time step
            list[list[float]]: the new empirical distributions over these
                particles
        """
        # nothing on the beams should be terminated
        # terminated paths should be in the terminated set
        for beam in beams:
            for path in beam:
                assert not path.terminated

        path_extensions = [[path.extend() for path in beam] for beam in beams]

        # for exploration, use a parser which pretends like every utterance
        # is the first utterance it is seeing
        ignore_previous_utterances = \
            self._config.independent_utterance_exploration

        # Use the ParseModel to score
        self._decoder.parse_model.score(flatten(path_extensions),
                                        ignore_previous_utterances,
                                        self._decoder.caching)

        new_beams = []
        new_distributions = []
        gamma = self._config.exploration_gamma
        for terminated_set, cases, distribution in zip(
                terminated, path_extensions, empirical_distributions):

            new_path_log_probs = []
            paths_to_sample_from = []

            for case, path_prob in zip(cases, distribution):
                for continuation in case.valid_continuations(
                        self._decoder.path_checker):
                    # Add all the terminated paths
                    if continuation.terminated:
                        terminated_set.add(continuation)
                    else:
                        # Sample from unterminated paths
                        new_path_log_probs.append(
                                gamma * continuation[-1].log_prob +
                                np.log(path_prob))
                        paths_to_sample_from.append(continuation)

            if len(paths_to_sample_from) == 0:
                new_beams.append([])
                new_distributions.append([])
                continue

            new_path_probs = softmax(new_path_log_probs)

            new_particles, new_distribution = self._sample(
                    paths_to_sample_from, new_path_probs)
            new_beams.append(new_particles)
            new_distributions.append(new_distribution)

        return terminated, new_beams, new_distributions

    @abstractmethod
    def _sample(self, paths_to_sample_from, path_probs):
        """Sample from set of valid paths to sample from according to policy.

        Args:
            paths_to_sample_from (list[ParsePath]): the valid paths in
                next beam
            path_probs (list[float]): gamma sharpened probs of each path

        Returns:
            list[ParsePath]: the paths that are sampled according to
                this policy
            list[float]: the new probabilities associated with these paths
                for the next iteration
        """
        raise NotImplementedError


################################
# Particle filter

class ParticleFiltering(GammaSamplingExplorationPolicy):
    """Estimates an empirical distribution from gamma-sharpened distribution
    given by ParseModel. Samples from that empirical distribution.

    1. Sample from empirical distribution p_hat (until get beam_size unique)
    2. Extend particles using true distribution

    Args:
        decoder (Decoder)
        config (Config)
        normalization (NormalizationOptions)
    """
    def _sample(self, paths_to_sample_from, path_probs):
        # Samples without replacement. New particles have empirical
        # distribution according to their frequency.
        num_to_sample = min(
                self._config.beam_size, len(paths_to_sample_from))
        sampled_particles = sample_with_replacement(
                paths_to_sample_from, path_probs, num_to_sample)
        new_particle_counts = Counter(sampled_particles)
        new_particles = list(new_particle_counts.keys())
        new_distribution = np.array(list(new_particle_counts.values()))
        new_distribution = list(
                new_distribution / float(np.sum(new_distribution)))
        return new_particles, new_distribution


################################
# Gamma Randomized Search

class GammaRandomizedSearch(GammaSamplingExplorationPolicy):
    def _sample(self, paths_to_sample_from, path_probs):
        # Samples without replacement
        num_to_sample = min(
                self._config.beam_size, len(paths_to_sample_from),
                sum(p > 0 for p in path_probs)
                )
        chosen_indices = np.random.choice(
                range(len(paths_to_sample_from)), size=num_to_sample,
                replace=False, p=path_probs)
        new_particles = [
                paths_to_sample_from[index] for index in chosen_indices]

        # Distribution is just gamma sharpened and normalized path probs
        new_distribution = softmax(
                [self._config.exploration_gamma * path.log_prob
                    for path in new_particles])
        return new_particles, new_distribution


################################
# Batched REINFORCE

class BatchedReinforce(ExplorationPolicy):
    """Exploration policy that sample K independent paths for each example
    (where K = beam size).
    
    - The paths comes from the model distribution p(z|x) with possible modifications
      using gamma or epsilon.
    - Specifically the next predicate is sampled from
        * gamma-softmaxed p(choice) with probability 1 - epsilon
        * uniform over choices      with probability epsilon
    - Choices that cannot be executed are not considered.
    - Paths that cannot be extended are discarded by default.
        * Turn on "zombie_mode" to keep them on the beam for negative update
    - There are two ways to handle terminated paths:
        * Default: The last predicate must be sampled like other predicates
        * termination_lookahead: For any choice that terminates the path,
            apply it and add the terminated path to the beam.
            Still keep extending the original path.

    Possible configs:
    - beam_size (int)
    - independent_utterance_exploration (bool)
    - exploration_gamma (float)
    - exploration_epsilon (float)
    - termination_lookahead (bool)
    - zombie_mode (bool)
    """

    def __init__(self, decoder, config, normalization, train):
        if not train:
            raise ValueError(
                    "Batched REINFORCE should only be used at train time")
        super(BatchedReinforce, self).__init__(
                decoder, config, normalization, train)

    def get_beams(self, examples, verbose=False):
        return self.get_intermediate_beams(examples, verbose)[0]

    def get_intermediate_beams(self, examples, verbose=False):
        # Start with beam_size empty paths for each example
        beams = [Beam([ParsePath.empty(ex.context)
                       for _ in range(self._config.beam_size)])
                for ex in examples]
        intermediates = [[] for _ in examples]
        num_iterations = get_num_iterations(
                self._config.iterations_per_utterance, examples)
        if verbose:
            iterations = verboserate(list(range(num_iterations)),
                                     desc='Batched REINFORCE')
        else:
            iterations = range(num_iterations)
        for _ in iterations:
            for ex_idx, beam in enumerate(beams):
                intermediates[ex_idx].append(beam)
            beams = self.advance(beams)
        return beams, intermediates

    def advance(self, beams):
        """Advance a batch of beams.

        Args:
            beams (list[Beam]): a batch of beams

        Returns:
            list[Beam]: a new batch of beams
                (in the same order as the input beams)
        """
        # Extend a new case for each unterminated path
        cases_to_be_scored = []
        extending = []
        for beam in beams:
            terminated, unterminated = [], []
            for path in beam:
                if path.terminated:
                    terminated.append(path)
                else:
                    case = path.extend()
                    cases_to_be_scored.append(case)
                    unterminated.append((path, case))
            extending.append((terminated, unterminated))
        # Score them
        ignore_previous_utterances = \
                self._config.independent_utterance_exploration
        self._decoder.parse_model.score(
                cases_to_be_scored, ignore_previous_utterances, False)
        # Read the scores and create new paths
        all_new_beams = []
        for new_beam, unterminated in extending:
            for old_path, case in unterminated:
                valid_choice_indices = []
                valid_new_paths = []
                for index, choice in enumerate(case.choices):
                    clone = case.copy_with_decision(choice)
                    denotation = clone.denotation
                    # Filter out the cases with invalid denotation
                    if not isinstance(denotation, Exception):
                        new_path = clone.path
                        # Filter out invalid paths
                        if new_path.terminated:
                            if new_path.finalizable:
                                # With termination_lookahead, add it to beam
                                if self._config.termination_lookahead:
                                    new_beam.append(new_path)
                                else:
                                    valid_choice_indices.append(index)
                                    valid_new_paths.append(new_path)
                        elif self._decoder.path_checker(new_path):
                            valid_choice_indices.append(index)
                            valid_new_paths.append(new_path)
                if valid_choice_indices:
                    # Sample a choice
                    epsilon = self._config.exploration_epsilon
                    gamma = self._config.exploration_gamma
                    if np.random.random() > epsilon:
                        probs = softmax([case.choice_logits[i] * gamma
                                         for i in valid_choice_indices])
                    else:
                        probs = ([1. / len(valid_choice_indices)]
                                * len(valid_choice_indices))
                    selected_index = np.random.choice(
                            list(range(len(valid_new_paths))), p=probs)
                    new_beam.append(valid_new_paths[selected_index])
                elif self._config.zombie_mode and len(old_path):
                    # Make a zombie copy of the last previous ParseCase
                    new_beam.append(old_path.zombie_clone())
            all_new_beams.append(Beam(new_beam))
        return all_new_beams


################################
# Main method

def get_exploration_policy(decoder, config, normalization, train):
    """Returns the ExplorationPolicy corresponding to the
    config.exploration_policy entry.

    Args:
        decoder (Decoder): The Decoder
        config (Config): Should be the config specified in the Decoder
        normalization (NormalizationOptions): The normalization
        train (bool): Whether the policy should be train or test

    Returns:
        ExplorationPolicy
    """
    if config.type == "beam-search":
        return BeamSearchExplorationPolicy(decoder, config, normalization, train)
    elif config.type == "particle-filtering":
        return ParticleFiltering(decoder, config, normalization, train)
    elif config.type == "gamma-randomized-search":
        return GammaRandomizedSearch(decoder, config, normalization, train)
    elif config.type == "stale-beam-search":
        return StaleBeamSearch(decoder, config, normalization, train)
    elif config.type == "batched-reinforce":
        return BatchedReinforce(decoder, config, normalization, train)
    else:
        raise ValueError(
                "{} does not specify a valid ExplorationPolicy".format(
                    config.type))
