"""Generate predicates based on the context (utterance + graph)
- FuzzyMatchGenerator:
    Generate predicates that fuzzily match an utterance span.
- NERValueGenerator:
    Generate predicates from NER values (numbers, dates, etc.)
    detected in the utterance
- FloatingPredicatesGenerator:
    Generate predicates that do not anchor to utterance spans
"""
import re
from abc import ABCMeta, abstractmethod

import Levenshtein

from strongsup.predicate import Predicate
from strongsup.predicates_computer import PredicatesComputer
from strongsup.tables.graph import NULL_CELL
from strongsup.tables.predicate import WikiTablePredicate, FIXED_PREDICATES


class Generator(object, metaclass=ABCMeta):
    @abstractmethod
    def get_predicates(self, tokens):
        """Return a dict mapping each matched predicate z to a list M[z]
        where M[z][i] indicates how well predicate z matches input token x[i].

        Args:
            tokens: A list of tokens of the utterance.
        """
        pass


################################
# FuzzyMatchGenerator

class PredicateInfo(object):
    def __init__(self, predicate, original_string, partial_match=False):
        self.predicate = predicate
        self.original_string = str(original_string)
        self.normalized_string = re.sub('[^a-z0-9]', '', original_string.lower())
        if partial_match:
            self.partials = set()
            if len(original_string) > FuzzyMatchGenerator.PARTIAL_MAX_PREDICATE_LENGTH:
                return
            tokens = re.sub('[^a-z0-9]', ' ', original_string.lower()).strip().split()
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens) + 1):
                    self.partials.add(''.join(tokens[i:j]))

class FuzzyMatchGenerator(Generator):
    """Compute possible string (or substring) matches between predicates
    from the graph and token spans from the utterance.
    """
    # Possible options:
    # - Do not consider the match if the score is less than this:
    SIMILARITY_THRESHOLD = 0.8
    # - Do not consider phrases matching > this number of predicates:
    #     (likely a stop word)
    MAX_MATCHES = 5
    # For partial matches (match a part of the predicate's string)
    # - Do not do partial match if the token span is shorter than this:
    PARTIAL_MIN_SPAN_LENGTH = 3
    # - Do not do partial match for predicates longer than this:
    PARTIAL_MAX_PREDICATE_LENGTH = 70

    def __init__(self, predicate_to_strings, partial_match=False):
        """Create a new TablesFuzzyMatcher for the given set of predicates.

        Args:
            predicate_to_strings: A dict from each predicate name
                (e.g., fb:cell.dr_who) to the original string ("Dr. Who")
            partial_match: Whether to perform partial match
                (allows the token span to match consecutive words from the predicate:
                 e.g., allows fb:cell.dr_who to match "who")
        """
        self.partial_match = partial_match
        self.predicate_infos = {}
        for predicate, original_string in predicate_to_strings.items():
            self.predicate_infos[predicate] = PredicateInfo(
                    predicate, original_string, partial_match)

    def get_predicates(self, tokens):
        """Return a dict mapping each fuzzy matched predicate z to a list M[z]
        where M[z][i] indicates how well predicate z matches input word x[i].

        Args:
            tokens: A list of string tokens.
        """
        tokens = [re.sub('[^a-z0-9]', '', token.lower()) for token in tokens]
        matches = {}
        for predicate_info in list(self.predicate_infos.values()):
            matches[predicate_info.predicate] = [0.0] * len(tokens)
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens) + 1):
                phrase = ''.join(tokens[i:j])
                predicates_matching_phrase = {}
                for predicate_info in list(self.predicate_infos.values()):
                    score = similarity_ratio(phrase, predicate_info.normalized_string)
                    if (self.partial_match and predicate_info.partials and
                            len(phrase) >= self.PARTIAL_MIN_SPAN_LENGTH):
                        part_score = max(similarity_ratio(phrase, part)
                                for part in predicate_info.partials)
                        score = max(score, part_score)
                    if score:
                        # Normalize
                        score = ((score - FuzzyMatchGenerator.SIMILARITY_THRESHOLD)
                                / (1. - FuzzyMatchGenerator.SIMILARITY_THRESHOLD))
                        predicates_matching_phrase[predicate_info.predicate] = score
                if len(predicates_matching_phrase) <= self.MAX_MATCHES:
                    for predicate, score in list(predicates_matching_phrase.items()):
                        weights = matches[predicate]
                        for k in range(i, j):
                            weights[k] = max(weights[k], score)
        return dict((predicate, weights)
                for (predicate, weights) in matches.items()
                if any(x > 0 for x in weights))


# Helper methods

def similarity_ratio(x, y, threshold=FuzzyMatchGenerator.SIMILARITY_THRESHOLD):
    """Compute the similarity ratio between two strings.
    If the ratio exceeds the threshold, return it; otherwise, return 0.

    The similarity ratio is given by
        1 - (levenshtein distance with substitution cost = 2) / (total length)
    """
    ratio = Levenshtein.ratio(x, y)
    return ratio if ratio > threshold else 0.


################################
# NERValueGenerator

class NERValueGenerator(Generator):
    """Compute possible primitive values (numbers and dates) that would be
    tagged with numerical and temporal NER classes (NUMBER, ORDINAL, DATE, etc.)

    Given an utterance x and a primitive value z, compute
        M[z][i] = how well predicate z matches input word x[i]
    """

    def __init__(self):
        pass

    def get_predicates(self, tokens):
        """Return a dict mapping each primitive predicate z to a list M[z]
        where M[z][i] indicates how well z matches input word x[i].

        Args:
            tokens: A list of string tokens.
        """
        match_weights = {}
        for i in range(len(tokens)):
            for predicate, l in hackish_ner(tokens, i):
                if predicate not in match_weights:
                    match_weights[predicate] = [0.0] * len(tokens)
                for k in range(i, i + l):
                    match_weights[predicate][k] = 1.0
        return match_weights


# Helper method
WORDS_CARDINAL = [
        'zero', 'one', 'two', 'three', 'four', 'five',
        'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
        'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
WORDS_ORDINAL = [
        'zeroth', 'first', 'second', 'third', 'fourth', 'fifth',
        'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
        'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
        'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth']
MONTHS = ['',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december']
MONTHS_SHORT = ['',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

def hackish_ner(tokens, i):
    """Check if tokens[i:i+l] can be converted into a primitive value.
    Yield pairs (predicate, length).
    """
    u = tokens[i]
    # Integers: 1, 20, 450, ...
    if re.match('^[0-9]+$', u):
        v = str(int(u))
        yield 'N' + v, 1
        if len(u) == 4:
            # Could be a year
            yield 'D{}-xx-xx'.format(v), 1
    # 0.2, 4,596, ...
    try:
        v = float(u.replace(',', ''))
        if int(v) == v:
            yield 'N' + str(int(v)), 1
        else:
            yield 'N' + str(v), 1
    except (ValueError, OverflowError):
        pass
    # 1st, 2nd, 3rd, ...
    match = re.match(r'([0-9]+)(st|nd|rd|th)', u)
    if match:
        yield 'N' + str(int(match.group(1))), 1
    # one, two, three, ...
    if u in WORDS_CARDINAL:
        yield 'N' + str(WORDS_CARDINAL.index(u)), 1
    # first, second, third, ...
    if u in WORDS_ORDINAL:
        yield 'N' + str(WORDS_ORDINAL.index(u)), 1
    # TODO: Handle dates and some more numbers
    # january, ...
    # march 6, march 6th, ...
    # november of 1992, november 1992, ...
    # "march 6 , 2012", ...
    # 1800s, 1920's, ...


################################
# FloatingPredicatesGenerator

class FloatingPredicatesGenerator(Generator):
    def __init__(self, graph):
        self.graph = graph

    def get_predicates(self, tokens):
        # Column headers
        match_weights = {}
        for predicate in self.graph._columns:
            match_weights[predicate] = None
            match_weights['!' + predicate] = None
        # Empty cell
        if self.graph.has_id(NULL_CELL):
            match_weights[NULL_CELL] = None
        return match_weights


################################
# Infer context predicates using all generator

class TablesPredicatesComputer(PredicatesComputer):
    NER_VALUE_GENERATOR = NERValueGenerator()

    def __init__(self, graph):
        """Initialize a predicates computer for the specified graph.

        Args:
            graph (TablesKnowledgeGraph)
        """
        self.graph = graph
        self.generators = [
                FloatingPredicatesGenerator(self.graph),
                FuzzyMatchGenerator(self.graph._original_strings, False),
                # The following one is kind of slow, so comment out for now.
                #FuzzyMatchGenerator(self.graph._original_strings, True),
                self.NER_VALUE_GENERATOR]

    def compute_predicates(self, tokens):
        """Infer predicates from the tokens and graph.
        Return a list of predicates.

        Args:
            tokens (list[unicode])
        Returns:
            list[(Predicate, alignment)]
            where alignment is list[(utterance token index, alignment strength)]
        """
        matches = []
        found_predicates = set()
        for matcher in self.generators:
            match_weights = matcher.get_predicates(tokens)
            matches.append(match_weights)
            found_predicates.update(match_weights)
        predicates = [(fixed, []) for fixed in FIXED_PREDICATES]
        for name in found_predicates:
            predicate = WikiTablePredicate(name,
                    original_string=self.get_original_string(name))
            match_weights = self.combine_match_weights(
                    matches, name, tokens)
            predicates.append((predicate, match_weights))
        predicates.sort(key=lambda x: (x[0].types, x[0].name))
        return predicates

    def get_original_string(self, name):
        """Get the original string. Also works for numbers and dates.

        Args:
            name (unicode): predicate name
        Returns:
            unicode
        """
        words = []
        if self.graph.has_id(name):
            words.append(self.graph.original_string(name))
        elif name[0] == 'N':
            words.append(name[1:])
        elif name[0] == 'D':
            year, month, day = name[1:].split('-')
            if year[0] != 'x':
                words.append(year)
            if month[0] != 'x':
                words.append(MONTHS[int(month)].title())
            if day[0] != 'x':
                words.append(day)
        return ' '.join(words)

    def combine_match_weights(self, matches, name, tokens):
        """Helper method for combining match weights.

        Returns:
            list[(utterance token index, alignment strength)]
        """
        combined = [0.0] * len(tokens)
        for match in matches:
            if name in match and match[name]:
                for i, x in enumerate(match[name]):
                    combined[i] = max(combined[i], x)
        return [(i, x) for (i, x) in enumerate(combined) if x]


################################
# Quick test

def test():
    from dependency.data_directory import DataDirectory
    from strongsup.tables.world import WikiTableWorld
    class DummyContext(object):
        def __init__(self, utterance, context):
            self.graph = WikiTableWorld(context).graph
            self.utterance = utterance
    with open(DataDirectory.wiki_table_questions + '/data/training.tsv') as fin:
        header = fin.readline().rstrip('\n').split('\t')
        for i in range(20):
            stuff = dict(list(zip(header, fin.readline().rstrip('\n').split('\t'))))
            context = DummyContext(stuff['utterance'].split(), stuff['context'])
            #print stuff
            predicates = TablesPredicatesComputer(context.graph).compute_predicates(context)
            names = [x[0].name for x in predicates]
            assert len(names) == len(set(names))
            #for x, y in sorted(predicates, key=lambda u: (u[0].types_vector, u[0].name)):
            #    print ' ' * 4, x, x.types, y

if __name__ == '__main__':
    test()
