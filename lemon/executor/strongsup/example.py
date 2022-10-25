from collections import Sequence

import sys

from gtd.io import JSONPicklable
from gtd.utils import cached_property, UnicodeMixin

from strongsup.predicate import Predicate
from strongsup.utils import PredicateList
from strongsup.value import Value
from strongsup.world import World


class Example(JSONPicklable):
    """An input context paired with the correct answer.

    Args:
        context (BaseContext)
        answer (list[Value]): target answer
        logical form (list [Predicate]): target logical form
    """
    def __init__(self, context, answer=None, logical_form=None):
        assert isinstance(context, BaseContext)
        self._context = context
        if answer:
            assert all(isinstance(x, Value) for x in answer)
        self._answer = answer
        if logical_form:
            assert all(isinstance(x, Predicate) for x in logical_form)
        self._logical_form = logical_form

    @property
    def context(self):
        return self._context

    @property
    def answer(self):
        """The correct answer to the question, as a list of Values.

        Returns:
            list[Value]
        """
        return self._answer

    @property
    def logical_form(self):
        """The correct logical form for the example.

        A list of Predicates.

        Raises:
            AttributeError, if no logical form present

        Returns:
            list[Predicate]
        """
        return self._logical_form

    def __getstate__(self):
        return self.context, self.answer, self.logical_form

    def __setstate__(self, state):
        context, answer, logical_form = state
        self.__init__(context, answer, logical_form)


class Utterance(Sequence, UnicodeMixin):
    __slots__ = ['_tokens', '_context', '_utterance_idx', '_predicates', '_predicate_alignments']

    def __init__(self, tokens, context, utterance_idx, predicate_alignments):
        """Create an Utterance.

        Args:
            tokens (tuple[unicode] | list[unicode]): list of words
            context (Context): context that this utterance belongs to
            utterance_idx (int): index of this utterance in context.utterances
            predicate_alignments (dict[Predicate, list[(int, float)]]): a map from predicates to alignments.
        """
        assert isinstance(tokens, list) or isinstance(tokens, tuple)
        if len(tokens) > 0:
            assert isinstance(tokens[0], str)
        self._tokens = tokens
        self._context = context
        self._utterance_idx = utterance_idx

        # compute allowable predicates and their alignments with the utterance
        self._predicate_alignments = predicate_alignments
        self._predicates = PredicateList(sorted(self._predicate_alignments.keys()))

    def __getitem__(self, i):
        return self._tokens[i]

    def __len__(self):
        return len(self._tokens)

    @property
    def context(self):
        return self._context

    @property
    def utterance_idx(self):
        return self._utterance_idx

    @property
    def _id(self):
        """An ID that uniquely identifies the utterance"""
        return (self.context, self.utterance_idx)

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        return other._id == self._id

    def __unicode__(self):
        return ' '.join(self._tokens)

    @property
    def predicates(self):
        """All allowable predicates for this utterance.

        CandidateGenerator uses this to generate candidates

        Returns:
            PredicateList (similar to list[Predicate] but with fast index lookup)
        """
        return self._predicates

    @property
    def predicate_alignments(self):
        return self._predicate_alignments

    def predicate_alignment(self, predicate):
        """Return the alignment between the specified predicate and utterance (for soft copying)

        Args:
            predicate (Predicate)
        Returns:
            list[(utterance token index, alignment strength)]
                utterance token index is an int in range(len(utterance))
                alignment strength is a float between 0 and 1, inclusive
        """
        if predicate not in self._predicate_alignments:
            #print >> sys.stderr, u'WARNING: {} not in matched predicates! [{}; {}]'.format(
            #        predicate, u' '.join(self._tokens), self.context.world)
            return []
        return self._predicate_alignments[predicate]


class DelexicalizedUtterance(Utterance):
    __slots__ = ['_placeholder_positions']

    def __init__(self, tokens, context, utterance_idx, predicate_alignments, placeholder_positions, orig_utterance):
        self._placeholder_positions = placeholder_positions
        self._original_utterance = orig_utterance
        super(DelexicalizedUtterance, self).__init__(tokens, context, utterance_idx, predicate_alignments)

    @property
    def original_utterance(self):
        return self._original_utterance

    @property
    def placeholder_positions(self):
        """A dict mapping from a Predicate to the list of positions in the delex'd utterance where it appears.

        Returns:
            dict[Predicate, list[int]]
        """
        return self._placeholder_positions


################################
# Context

class BaseContext(UnicodeMixin):
    def __init__(self, world, utterances):
        """Initialize a Context.

        Args:
            world (World)
            utterances (list[Utterance])
        """
        assert isinstance(world, World)
        self._world = world
        self._utterances = utterances

        # aggregate predicates
        preds_union = set()
        for utt in self._utterances:
            preds_union.update(utt.predicates)
        self._predicates = PredicateList(sorted(preds_union))

        self._silver_logical_form = None

    @property
    def world(self):
        """Return the World."""
        return self._world

    @property
    def utterances(self):
        """Utterances.

        Returns:
            list[Utterance]
        """
        return self._utterances

    @property
    def predicates(self):
        """The union of the allowable predicates for each utterance in this context.

        CandidateGenerator uses this to generate candidates.

        Returns:
            PredicateList (similar to list[Predicate] but with fast index lookup)
        """
        return self._predicates

    @property
    def silver_logical_form(self):
        """Parse path for highest prob logical form that has been generated
        for this context that executes to the correct denotation. Could
        be None.

        Returns:
            ParsePath
        """
        return self._silver_logical_form

    @property
    def executor(self):
        """Return the Executor."""
        return self._world.executor

    def __unicode__(self):
        return '\n'.join([str(utt) for utt in self.utterances])


class Context(BaseContext):
    """The necessary and sufficient information to answer a query utterance."""

    def __init__(self, world, raw_utterances):
        """Initialize a Context.

        Args:
            world (World)
            raw_utterances (list[list[unicode]])
        """
        assert isinstance(raw_utterances, list), raw_utterances
        assert isinstance(raw_utterances[0], list), raw_utterances[0]
        assert isinstance(raw_utterances[0][0], str), raw_utterances[0][0]

        # compute Predicate alignments and construct Utterance objects
        utterances = []
        for i, raw_utt in enumerate(raw_utterances):
            predicate_alignments = dict(world.predicates_computer.compute_predicates(raw_utt))
            utt = Utterance(raw_utt, self, i, predicate_alignments)
            utterances.append(utt)

        super(Context, self).__init__(world, utterances)


class DelexicalizedContext(BaseContext):
    def __init__(self, context):
        self._original_context = context
        utterances = context.utterances
        delex_utterances = [self._delexicalize_utterance(utt) for utt in utterances]
        super(DelexicalizedContext, self).__init__(context.world, delex_utterances)

    @property
    def original_context(self):
        return self._original_context

    def _delexicalize_utterance(self, utt):
        """Compute the delexicalized version of the utterance.

        Args:
            utt (Utterance): the original utterance

        Some phrases are collapsed into placeholders strings.
        These strings are derived from predicate.delexicalized_name
        and conventionally begin with an uppercase letter.

        Delexicalization uses this strategy:
            - Sort aligned predicates by score (sum of alignment weights)
            - Starting from higher scores, mark out the utterance tokens
              that each predicate is aligned to.

        The set of predicates on the utterance remain the same.

        The predicate alignment positions are now relative
        to the new delexicalized utterance. Alignment strengths
        to the collapsed tokens are averaged out.
        """
        if isinstance(utt, DelexicalizedUtterance):
            raise ValueError('Already delexicalized.')

        # Sort the predicates by heuristic scores
        aligned_predicates = []    # (predicate, alignment, score)
        for predicate, alignment in utt.predicate_alignments.items():
            # Ignore some predicates (unaligned or should not be delexicalized)
            if not alignment or predicate.delexicalized_name is None:
                continue
            # Compute the clean alignment (only use the exact-matched portion)
            clean_alignment = [
                    index for (index, strength) in alignment
                    if strength == 1.0]
            # Cut into contiguous segments
            clean_segments = []
            for x in clean_alignment:
                if not clean_segments or x != clean_segments[-1][-1] + 1:
                    clean_segments.append([x])
                else:
                    clean_segments[-1].append(x)
            #score = sum(strength for (_, strength) in alignment)
            for segment in clean_segments:
                aligned_predicates.append((predicate, segment, len(segment)))
        aligned_predicates.sort(key=lambda x: -x[2])
        # Greedily replace utterance tokens with placeholders
        replacements = [False] * len(utt)
        for predicate, segment, score in aligned_predicates:
            # Avoid overlap
            if any(replacements[index] for index in segment):
                continue
            for index in segment:
                replacements[index] = predicate
        # Compute the delexicalized utterance
        tokens = []
        placeholder_positions = {}
        old_to_new_indices = []
        last_replacement = None
        for token, replacement in zip(utt, replacements):
            if not replacement:
                tokens.append(token)
            elif replacement != last_replacement:
                placeholder_positions\
                    .setdefault(replacement, []).append(len(tokens))
                tokens.append(replacement.delexicalized_name)
            old_to_new_indices.append(len(tokens) - 1)
            last_replacement = replacement
        # Compute predicate_alignments
        predicate_alignments = {}
        for predicate, old_alignment in utt.predicate_alignments.items():
            if not old_alignment:
                predicate_alignments[predicate] = old_alignment
            else:
                new_alignment = {}
                for index, strength in old_alignment:
                    new_index = old_to_new_indices[index]
                    new_alignment.setdefault(new_index, []).append(strength)
                predicate_alignments[predicate] = [
                        (index, sum(strengths) / len(strengths))
                        for (index, strengths) in new_alignment.items()]
            # Add placeholder positions for reversed relations
            if predicate.name[0] == '!':
                for x in placeholder_positions:
                    if x.name == predicate.name[1:]:
                        placeholder_positions[predicate] = \
                                placeholder_positions[x]
                        break

        return DelexicalizedUtterance(tokens, self, utt.utterance_idx, predicate_alignments,
                                      placeholder_positions, utt)
