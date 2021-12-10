#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
It also knows all the standard test sets and handles downloading, processing, and tokenization for you.

See the [README.md] file for more information.
"""

import io
import math
import sys
import logging
import pathlib
import argparse
from collections import Counter

import sacrebleu
# Allows calling the script as a standalone utility
# See: https://github.com/mjpost/sacrebleu/issues/86
from sacrebleu.metrics.bleu import BLEUSignature
from typing import List, Iterable, Union

if sacrebleu.__package__ is None and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'sacrebleu'

from sacrebleu.tokenizers import TOKENIZERS, DEFAULT_TOKENIZER
from sacrebleu.dataset import DATASETS, DOMAINS, COUNTRIES, SUBSETS
from sacrebleu.metrics import METRICS

from sacrebleu.utils import smart_open, filter_subset, get_available_origlangs, SACREBLEU_DIR, my_log
from sacrebleu.utils import get_langpairs_for_testset, get_available_testsets
from sacrebleu.utils import print_test_set, get_reference_files, download_test_set
from sacrebleu import __version__ as VERSION

sacrelogger = logging.getLogger('sacrebleu')

class BLEU2:
    NGRAM_ORDER = 4

    SMOOTH_DEFAULTS = {
        # The defaults for `floor` and `add-k` are obtained from the following paper
        # A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU
        # Boxing Chen and Colin Cherry
        # http://aclweb.org/anthology/W14-3346
        'none': None,   # No value is required
        'floor': 0.1,
        'add-k': 1,
        'exp': None,    # No value is required
    }

    def __init__(self, args):
        self.name = 'bleu'
        self.force = args.force
        self.lc = args.lc
        self.smooth_value = args.smooth_value
        self.smooth_method = args.smooth_method
        self.tokenizer = TOKENIZERS[args.tokenize]()
        self.signature = BLEUSignature(args)

        # Sanity check
        assert self.smooth_method in self.SMOOTH_DEFAULTS.keys(), \
            "Unknown smooth_method '{}'".format(self.smooth_method)

    @staticmethod
    def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
        """Extracts all the ngrams (min_order <= n <= max_order) from a sequence of tokens.

        :param line: A segment containing a sequence of words.
        :param min_order: Minimum n-gram length (default: 1).
        :param max_order: Maximum n-gram length (default: NGRAM_ORDER).
        :return: a dictionary containing ngrams and counts
        """

        ngrams = Counter()  # type: Counter
        tokens = line.split()
        for n in range(min_order, max_order + 1):
            for i in range(0, len(tokens) - n + 1):
                ngram = ' '.join(tokens[i: i + n])
                ngrams[ngram] += 1

        return ngrams

    @staticmethod
    def reference_stats(refs, output_len):
        """Extracts reference statistics for a given segment.

        :param refs: A list of segment tokens.
        :param output_len: Hypothesis length for this segment.
        :return: a tuple of (ngrams, closest_diff, closest_len)
        """

        ngrams = Counter()
        closest_diff = None
        closest_len = None

        for ref in refs:
            tokens = ref.split()
            reflen = len(tokens)
            diff = abs(output_len - reflen)
            if closest_diff is None or diff < closest_diff:
                closest_diff = diff
                closest_len = reflen
            elif diff == closest_diff:
                if reflen < closest_len:
                    closest_len = reflen

            ngrams_ref = BLEU2.extract_ngrams(ref)
            for ngram in ngrams_ref.keys():
                ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

        return ngrams, closest_diff, closest_len

    @staticmethod
    def compute_bleu(correct: List[int],
                     total: List[int],
                     sys_len: int,
                     ref_len: int,
                     smooth_method: str = 'none',
                     smooth_value=None,
                     use_effective_order=False) -> sacrebleu.BLEUScore:
        """Computes BLEU score from its sufficient statistics. Adds smoothing.

        Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
        Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

        - none: No smoothing.
        - floor: Method 1 (requires small positive value (0.1 in the paper) to be set)
        - add-k: Method 2 (Generalizing Lin and Och, 2004)
        - exp: Method 3 (NIST smoothing method i.e. in use with mteval-v13a.pl)

        :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
        :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
        :param sys_len: The cumulative system length
        :param ref_len: The cumulative reference length
        :param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none')
        :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
        :param use_effective_order: If true, use the length of `correct` for the n-gram order instead of NGRAM_ORDER.
        :return: A BLEU object with the score (100-based) and other statistics.
        """
        assert smooth_method in BLEU2.SMOOTH_DEFAULTS.keys(), \
            "Unknown smooth_method '{}'".format(smooth_method)

        # Fetch the default value for floor and add-k
        if smooth_value is None:
            smooth_value = BLEU2.SMOOTH_DEFAULTS[smooth_method]

        precisions = [0.0 for x in range(BLEU2.NGRAM_ORDER)]

        smooth_mteval = 1.
        effective_order = BLEU2.NGRAM_ORDER
        for n in range(1, BLEU2.NGRAM_ORDER + 1):
            if smooth_method == 'add-k' and n > 1:
                correct[n-1] += smooth_value
                total[n-1] += smooth_value
            if total[n-1] == 0:
                break

            if use_effective_order:
                effective_order = n

            if correct[n-1] == 0:
                if smooth_method == 'exp':
                    smooth_mteval *= 2
                    precisions[n-1] = 100. / (smooth_mteval * total[n-1])
                elif smooth_method == 'floor':
                    precisions[n-1] = 100. * smooth_value / total[n-1]
            else:
                precisions[n-1] = 100. * correct[n-1] / total[n-1]

        # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU
        # score is 0 (technically undefined). This is a problem for sentence
        # level BLEU or a corpus of short sentences, where systems will get
        # no credit if sentence lengths fall under the NGRAM_ORDER threshold.
        # This fix scales NGRAM_ORDER to the observed maximum order.
        # It is only available through the API and off by default

        if sys_len < ref_len:
            bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
        else:
            bp = 1.0

        score = bp * math.exp(
            sum(map(my_log, precisions[:effective_order])) / effective_order)

        return sacrebleu.BLEUScore(
            score, correct, total, precisions, bp, sys_len, ref_len)

    def sentence_score(self, hypothesis: str,
                       references: List[str],
                       use_effective_order: bool = True) -> sacrebleu.BLEUScore:
        """
        Computes BLEU on a single sentence pair.

        Disclaimer: computing BLEU on the sentence level is not its intended use,
        BLEU is a corpus-level metric.

        :param hypothesis: Hypothesis string.
        :param references: List of reference strings.
        :param use_effective_order: Account for references that are shorter than the largest n-gram.
        :return: a `BLEUScore` object containing everything you'd want
        """
        assert not isinstance(references, str), \
            "sentence_score needs a list of references, not a single string"
        return self.corpus_score(hypothesis, [[ref] for ref in references],
                                 use_effective_order=use_effective_order)

    def corpus_score(self, sys_stream: Union[str, Iterable[str]],
                     ref_streams: Union[str, List[Iterable[str]]],
                     use_effective_order: bool = False) -> sacrebleu.BLEUScore:
        """Produces BLEU scores along with its sufficient statistics from a source against one or more references.

        :param sys_stream: The system stream (a sequence of segments)
        :param ref_streams: A list of one or more reference streams (each a sequence of segments)
        :param use_effective_order: Account for references that are shorter than the largest n-gram.
        :return: a `BLEUScore` object containing everything you'd want
        """

        # Add some robustness to the input arguments
        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]

        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        sys_len = 0
        ref_len = 0

        correct = [0 for n in range(self.NGRAM_ORDER)]
        total = [0 for n in range(self.NGRAM_ORDER)]

        # look for already-tokenized sentences
        tokenized_count = 0

        # sanity checks
        if any(len(ref_stream) != len(sys_stream) for ref_stream in ref_streams):
            raise EOFError("System and reference streams have different lengths!")
        if any(line is None for line in sys_stream):
            raise EOFError("Undefined line in system stream!")

        for output, *refs in zip(sys_stream, *ref_streams):
            # remove undefined/empty references (i.e. we have fewer references for this particular sentence)
            # but keep empty hypothesis (it's always defined thanks to the sanity check above)
            lines = [output] + [x for x in refs if x is not None and x != ""]
            if len(lines) < 2:  # we need at least hypothesis + 1 defined & non-empty reference
                raise EOFError("No valid references for a sentence!")

            if self.lc:
                lines = [x.lower() for x in lines]

            if not (self.force or self.tokenizer.signature() == 'none') and lines[0].rstrip().endswith(' .'):
                tokenized_count += 1

                if tokenized_count == 100:
                    sacrelogger.warning('That\'s 100 lines that end in a tokenized period (\'.\')')
                    sacrelogger.warning('It looks like you forgot to detokenize your test data, which may hurt your score.')
                    sacrelogger.warning('If you insist your data is detokenized, or don\'t care, you can suppress this message with \'--force\'.')

            output, *refs = [self.tokenizer(x.rstrip()) for x in lines]

            output_len = len(output.split())
            ref_ngrams, closest_diff, closest_len = BLEU2.reference_stats(refs, output_len)

            sys_len += output_len
            ref_len += closest_len

            sys_ngrams = BLEU2.extract_ngrams(output)
            for ngram in sys_ngrams.keys():
                n = len(ngram.split())
                correct[n-1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
                total[n-1] += sys_ngrams[ngram]

        # Get BLEUScore object
        score = self.compute_bleu(
            correct, total, sys_len, ref_len,
            smooth_method=self.smooth_method, smooth_value=self.smooth_value,
            use_effective_order=use_effective_order)

        return score

METRICS['bleu'] = BLEU2
try:
    # SIGPIPE is not available on Windows machines, throwing an exception.
    from signal import SIGPIPE

    # If SIGPIPE is available, change behaviour to default instead of ignore.
    from signal import signal, SIG_DFL
    signal(SIGPIPE, SIG_DFL)

except ImportError:
    sacrelogger.warning('Could not import signal.SIGPIPE (this is expected on Windows machines)')


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='sacreBLEU: Hassle-free computation of shareable BLEU scores.\n'
                    'Quick usage: score your detokenized output against WMT\'14 EN-DE:\n'
                    '    cat output.detok.de | sacrebleu -t wmt14 -l en-de',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    arg_parser.add_argument('--citation', '--cite', default=False, action='store_true',
                            help='dump the bibtex citation and quit.')
    arg_parser.add_argument('--list', default=False, action='store_true',
                            help='print a list of all available test sets.')
    arg_parser.add_argument('--test-set', '-t', type=str, default=None,
                            help='the test set to use (see also --list) or a comma-separated list of test sets to be concatenated')
    arg_parser.add_argument('--language-pair', '-l', dest='langpair', default=None,
                            help='source-target language pair (2-char ISO639-1 codes)')
    arg_parser.add_argument('--origlang', '-ol', dest='origlang', default=None,
                            help='use a subset of sentences with a given original language (2-char ISO639-1 codes), "non-" prefix means negation')
    arg_parser.add_argument('--subset', dest='subset', default=None,
                            help='use a subset of sentences whose document annotation matches a give regex (see SUBSETS in the source code)')
    arg_parser.add_argument('--download', type=str, default=None,
                            help='download a test set and quit')
    arg_parser.add_argument('--echo', choices=['src', 'ref', 'both'], type=str, default=None,
                            help='output the source (src), reference (ref), or both (both, pasted) to STDOUT and quit')

    # I/O related arguments
    arg_parser.add_argument('--input', '-i', type=str, default='-',
                            help='Read input from a file instead of STDIN')
    arg_parser.add_argument('refs', nargs='*', default=[],
                            help='optional list of references (for backwards-compatibility with older scripts)')
    arg_parser.add_argument('--num-refs', '-nr', type=int, default=1,
                            help='Split the reference stream on tabs, and expect this many references. Default: %(default)s.')
    arg_parser.add_argument('--encoding', '-e', type=str, default='utf-8',
                            help='open text files with specified encoding (default: %(default)s)')

    # Metric selection
    arg_parser.add_argument('--metrics', '-m', choices=METRICS.keys(), nargs='+', default=['bleu'],
                            help='metrics to compute (default: bleu)')
    arg_parser.add_argument('--sentence-level', '-sl', action='store_true', help='Output metric on each sentence.')

    # BLEU-related arguments
    arg_parser.add_argument('-lc', action='store_true', default=False, help='Use case-insensitive BLEU (default: False)')
    arg_parser.add_argument('--smooth-method', '-s', choices=METRICS['bleu'].SMOOTH_DEFAULTS.keys(), default='exp',
                            help='smoothing method: exponential decay (default), floor (increment zero counts), add-k (increment num/denom by k for n>1), or none')
    arg_parser.add_argument('--smooth-value', '-sv', type=float, default=None,
                            help='The value to pass to the smoothing technique, only used for floor and add-k. Default floor: {}, add-k: {}.'.format(
                                METRICS['bleu'].SMOOTH_DEFAULTS['floor'], METRICS['bleu'].SMOOTH_DEFAULTS['add-k']))
    arg_parser.add_argument('--tokenize', '-tok', choices=TOKENIZERS.keys(), default=None,
                            help='Tokenization method to use for BLEU. If not provided, defaults to `zh` for Chinese, `mecab` for Japanese and `mteval-v13a` otherwise.')
    arg_parser.add_argument('--force', default=False, action='store_true',
                            help='insist that your tokenized input is actually detokenized')

    # ChrF-related arguments
    arg_parser.add_argument('--chrf-order', type=int, default=METRICS['chrf'].ORDER,
                            help='chrf character order (default: %(default)s)')
    arg_parser.add_argument('--chrf-beta', type=int, default=METRICS['chrf'].BETA,
                            help='chrf BETA parameter (default: %(default)s)')
    arg_parser.add_argument('--chrf-whitespace', action='store_true', default=False,
                            help='include whitespace in chrF calculation (default: %(default)s)')

    # Reporting related arguments
    arg_parser.add_argument('--quiet', '-q', default=False, action='store_true',
                            help='suppress informative output')
    arg_parser.add_argument('--short', default=False, action='store_true',
                            help='produce a shorter (less human readable) signature')
    arg_parser.add_argument('--score-only', '-b', default=False, action='store_true',
                            help='output only the BLEU score')
    arg_parser.add_argument('--width', '-w', type=int, default=1,
                            help='floating point width (default: %(default)s)')
    arg_parser.add_argument('--detail', '-d', default=False, action='store_true',
                            help='print extra information (split test sets based on origlang)')

    arg_parser.add_argument('-V', '--version', action='version',
                            version='%(prog)s {}'.format(VERSION))
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()

    # Explicitly set the encoding
    sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='utf-8', buffering=True, newline="\n")
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=True)

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format='sacreBLEU: %(message)s')

    if args.download:
        download_test_set(args.download, args.langpair)
        sys.exit(0)

    if args.list:
        if args.test_set:
            print(' '.join(get_langpairs_for_testset(args.test_set)))
        else:
            print('The available test sets are:')
            for testset in get_available_testsets():
                print('%30s: %s' % (testset, DATASETS[testset].get('description', '').strip()))
        sys.exit(0)

    if args.sentence_level and len(args.metrics) > 1:
        sacrelogger.error('Only one metric can be used with Sentence-level reporting.')
        sys.exit(1)

    if args.citation:
        if not args.test_set:
            sacrelogger.error('I need a test set (-t).')
            sys.exit(1)
        for test_set in args.test_set.split(','):
            if 'citation' not in DATASETS[test_set]:
                sacrelogger.error('No citation found for %s', test_set)
            else:
                print(DATASETS[test_set]['citation'])
        sys.exit(0)

    if args.num_refs != 1 and (args.test_set is not None or len(args.refs) > 1):
        sacrelogger.error('The --num-refs argument allows you to provide any number of tab-delimited references in a single file.')
        sacrelogger.error('You can only use it with externaly-provided references, however (i.e., not with `-t`),')
        sacrelogger.error('and you cannot then provide multiple reference files.')
        sys.exit(1)

    if args.test_set is not None:
        for test_set in args.test_set.split(','):
            if test_set not in DATASETS:
                sacrelogger.error('Unknown test set "%s"', test_set)
                sacrelogger.error('Please run with --list to see the available test sets.')
                sys.exit(1)

    if args.test_set is None:
        if len(args.refs) == 0:
            sacrelogger.error('I need either a predefined test set (-t) or a list of references')
            sacrelogger.error(get_available_testsets())
            sys.exit(1)
    elif len(args.refs) > 0:
        sacrelogger.error('I need exactly one of (a) a predefined test set (-t) or (b) a list of references')
        sys.exit(1)
    elif args.langpair is None:
        sacrelogger.error('I need a language pair (-l).')
        sys.exit(1)
    else:
        for test_set in args.test_set.split(','):
            langpairs = get_langpairs_for_testset(test_set)
            if args.langpair not in langpairs:
                sacrelogger.error('No such language pair "%s"', args.langpair)
                sacrelogger.error('Available language pairs for test set "%s": %s', test_set,
                                  ', '.join(langpairs))
                sys.exit(1)

    if args.echo:
        if args.langpair is None or args.test_set is None:
            sacrelogger.warning("--echo requires a test set (--t) and a language pair (-l)")
            sys.exit(1)
        for test_set in args.test_set.split(','):
            print_test_set(test_set, args.langpair, args.echo, args.origlang, args.subset)
        sys.exit(0)

    if args.test_set is not None and args.tokenize == 'none':
        sacrelogger.warning("You are turning off sacrebleu's internal tokenization ('--tokenize none'), presumably to supply\n"
                            "your own reference tokenization. Published numbers will not be comparable with other papers.\n")

    if 'ter' in args.metrics and args.tokenize is not None:
        logging.warning("Your setting of --tokenize will be ignored when "
                        "computing TER")

    # Internal tokenizer settings
    if args.tokenize is None:
        # set default
        if args.langpair is not None and args.langpair.split('-')[1] == 'zh':
            args.tokenize = 'zh'
        elif args.langpair is not None and args.langpair.split('-')[1] == 'ja':
            args.tokenize = 'ja-mecab'
        else:
            args.tokenize = DEFAULT_TOKENIZER

    if args.langpair is not None and 'bleu' in args.metrics:
        if args.langpair.split('-')[1] == 'zh' and args.tokenize != 'zh':
            sacrelogger.warning('You should also pass "--tok zh" when scoring Chinese...')
        if args.langpair.split('-')[1] == 'ja' and not args.tokenize.startswith('ja-'):
            sacrelogger.warning('You should also pass "--tok ja-mecab" when scoring Japanese...')

    # concat_ref_files is a list of list of reference filenames, for example:
    # concat_ref_files = [[testset1_refA, testset1_refB], [testset2_refA, testset2_refB]]
    if args.test_set is None:
        concat_ref_files = [args.refs]
    else:
        concat_ref_files = []
        for test_set in args.test_set.split(','):
            ref_files = get_reference_files(test_set, args.langpair)
            if len(ref_files) == 0:
                sacrelogger.warning('No references found for test set {}/{}.'.format(test_set, args.langpair))
            concat_ref_files.append(ref_files)

    # Read references
    full_refs = [[] for x in range(max(len(concat_ref_files[0]), args.num_refs))]
    for ref_files in concat_ref_files:
        for refno, ref_file in enumerate(ref_files):
            for lineno, line in enumerate(smart_open(ref_file, encoding=args.encoding), 1):
                if args.num_refs != 1:
                    splits = line.rstrip().split(sep='\t', maxsplit=args.num_refs-1)
                    if len(splits) != args.num_refs:
                        sacrelogger.error('FATAL: line {}: expected {} fields, but found {}.'.format(lineno, args.num_refs, len(splits)))
                        sys.exit(17)
                    for refno, split in enumerate(splits):
                        full_refs[refno].append(split)
                else:
                    full_refs[refno].append(line)

    # Decide on the number of final references, override the argument
    args.num_refs = len(full_refs)

    # Read hypotheses stream
    if args.input == '-':
        inputfh = io.TextIOWrapper(sys.stdin.buffer, encoding=args.encoding)
    else:
        inputfh = smart_open(args.input, encoding=args.encoding)
    full_system = inputfh.readlines()

    # Filter sentences according to a given origlang
    system, *refs = filter_subset(
        [full_system, *full_refs], args.test_set, args.langpair, args.origlang, args.subset)

    if len(system) == 0:
        message = 'Test set %s contains no sentence' % args.test_set
        if args.origlang is not None or args.subset is not None:
            message += ' with'
            message += '' if args.origlang is None else ' origlang=' + args.origlang
            message += '' if args.subset is None else ' subset=' + args.subset
        sacrelogger.error(message)
        sys.exit(1)

    # Create metric inventory, let each metric consume relevant args from argparse
    metrics = [METRICS[met](args) for met in args.metrics]

    # Handle sentence level and quit
    if args.sentence_level:
        # one metric in use for sentence-level
        metric = metrics[0]
        for output, *references in zip(system, *refs):
            score = metric.sentence_score(output, references)
            print(score.format(args.width, args.score_only, metric.signature))

        sys.exit(0)

    # Else, handle system level
    for metric in metrics:
        try:
            score = metric.corpus_score(system, refs)
        except EOFError:
            sacrelogger.error('The input and reference stream(s) were of different lengths.')
            if args.test_set is not None:
                sacrelogger.error('\nThis could be a problem with your system output or with sacreBLEU\'s reference database.\n'
                                  'If the latter, you can clean out the references cache by typing:\n'
                                  '\n'
                                  '    rm -r %s/%s\n'
                                  '\n'
                                  'They will be downloaded automatically again the next time you run sacreBLEU.', SACREBLEU_DIR,
                                  args.test_set)
            sys.exit(1)
        else:
            print(score.format(args.width, args.score_only, metric.signature))

    if args.detail:
        width = args.width
        sents_digits = len(str(len(full_system)))
        origlangs = args.origlang if args.origlang else get_available_origlangs(args.test_set, args.langpair)
        for origlang in origlangs:
            subsets = [None]
            if args.subset is not None:
                subsets += [args.subset]
            elif all(t in SUBSETS for t in args.test_set.split(',')):
                subsets += COUNTRIES + DOMAINS
            for subset in subsets:
                system, *refs = filter_subset([full_system, *full_refs], args.test_set, args.langpair, origlang, subset)
                if len(system) == 0:
                    continue
                if subset in COUNTRIES:
                    subset_str = '%20s' % ('country=' + subset)
                elif subset in DOMAINS:
                    subset_str = '%20s' % ('domain=' + subset)
                else:
                    subset_str = '%20s' % ''
                for metric in metrics:
                    # FIXME: handle this in metrics
                    if metric.name == 'bleu':
                        _refs = refs
                    elif metric.name == 'chrf':
                        _refs = refs[0]

                    score = metric.corpus_score(system, _refs)
                    print('origlang={} {}: sentences={:{}} {}={:{}.{}f}'.format(
                        origlang, subset_str, len(system), sents_digits,
                        score.prefix, score.score, width+4, width))


if __name__ == '__main__':
    main()