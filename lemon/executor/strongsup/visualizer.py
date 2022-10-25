import os
import re

from codecs import open

import itertools

from strongsup.example import DelexicalizedContext
from strongsup.evaluation import Evaluation, BernoulliSequenceStat
from strongsup.value import check_denotation
from strongsup.utils import EOU


class Visualizer(object):
    """Subclass around a Decoder, which does exactly the same thing as Decoder, but also
    prints out context and predictions.

    Args:
        decoder (Decoder)
        filename (unicode string): filename of where to write the output. This
            overwrites the file
        group_name (str): group name (show up in log and tensorboard)
        train (bool): If this is a train or a valid Visualizer
    """
    def __init__(self, decoder, filename, group_name, train):
        self._decoder = decoder
        self._filename = filename
        self._group_name = group_name
        self._train = train

    @property
    def group_name(self):
        return self._group_name

    def log_silver_logical_forms(self, examples):
        """Logs the silver logical forms of the examples if they exist.

        Args:
            examples (list[Example]): the examples
        """
        with open(self._filename, 'a+') as log:
            for example in examples:
                self._log_example_basic(example, log)

    def _log_example_basic(self, example, log):
        """Logs the basic info for a single example."""
        context = example.context
        log.write('Utterances:\n{}\n'.format(context).encode('utf-8'))

        if isinstance(context, DelexicalizedContext):
            log.write('Orig Utterances:\n{}\n'.format(context.original_context).encode('utf-8'))

        log.write('World: {}\n'.format(context.world))
        log.write('Denotation: {}\n'.format(example.answer))
        log.write('Gold Logical Form: {}\n'.format(example.logical_form))
        if example.logical_form:
            pattern = _logical_form_pattern(example.logical_form)
            log.write('Gold Logical Form Pattern: {}\n'.format(pattern))
        if example.context.silver_logical_form is not None:
            log.write('Silver Logical Form: {}\n\n'.format(
                example.context.silver_logical_form.decisions))
        else:
            log.write('Silver Logical Form: None\n\n')

    def reset(self, step=None):
        """Reset the output file and print the header."""
        with open(self._filename, 'w') as log:
            log.write('\n################ {}-{} ################\n\n'.format(step, self._group_name))

    def predictions(self, examples, verbose=False):
        """Gets predictions from decoder and prints out additional information

        Args:
            examples (list[Example]): a batch of Example

        Returns:
            list[Beam]: a batch of Beams
        """
        with open(self._filename, 'a') as log:
            contexts = [example.context for example in examples]
            beams, intermediates = self._decoder.get_intermediate_beams(
                    examples, train=self._train, verbose=verbose)
            evaluation = Evaluation()
            for beam, intermeds, example in zip(beams, intermediates, examples):
                self._log_beam(beam, intermeds, example, log, evaluation)
            return [beam.get_terminated() for beam in beams], evaluation

    def _log_beam(self, final_beam, intermediate_beams, example, log, evaluation,
            top_n=10, log_all_beams=True):
        """Takes a single prediction and logs information in a file

        Args:
            final_beam (Beam)
            intermediate_beams (list[Beam])
            example (Example): the example
            log (file): file to dump output to
            evaluation (Evaluation): statistics collector
            top_n (int): number of predictions to print out
            log_all_beams (bool): whether to log all intermediate beams
        """
        context = example.context
        predictions = final_beam.get_terminated()
        ranker = self._decoder.exploration_policy(self._train)._ranker
        probs = self._decoder.get_probs(predictions)
        evaluation.add('numCandidates_{}'.format(self._group_name), len(predictions))

        log.write('World:\n')
        context.world.dump_human_readable(log)
        log.write('\n')

        self._log_example_basic(example, log)

        first_candidate = 0 if predictions else None
        num_deno_corrects = 0
        first_deno_correct = None
        deno_hit_mass = 0.

        for i, prediction in enumerate(predictions):
            denotation = prediction.finalized_denotation
            is_correct = check_denotation(example.answer, denotation)
            if is_correct:
                log.write('Deno correct at {}\n'.format(i))
                num_deno_corrects += 1
                if first_deno_correct is None:
                    first_deno_correct = i
                deno_hit_mass += probs[i]
            if is_correct or i < top_n:
                log.write('Predicted Logical Form {}: {}\n'.format(i, prediction.decisions))
                log.write('Predicted Denotation {}: {}\n'.format(i, denotation))
        log.write('\n')

        if not predictions:
            log.write('No predictions\n')
        else:
            log.write('Candidates with correct denotation: {} / {}\n'.format(num_deno_corrects, len(predictions)))
            log.write('First deno correct: {} / {}\n'.format(first_deno_correct, len(predictions)))
        log.write('\n')

        # Denotation Evaluation
        deno_acc = (first_deno_correct == 0)
        evaluation.add('denoAcc_{}'.format(self._group_name), deno_acc)
        evaluation.add('denoHit_{}'.format(self._group_name), num_deno_corrects > 0)
        evaluation.add('denoSpu_{}'.format(self._group_name), num_deno_corrects)
        evaluation.add('denoHitMass_{}'.format(self._group_name), deno_hit_mass)

        # Separate by number of utterances
        num_utterances = len(context.utterances)
        evaluation.add('denoAcc_{}_{}utts'.format(self._group_name, num_utterances), deno_acc)
        evaluation.add('denoHit_{}_{}utts'.format(self._group_name, num_utterances), num_deno_corrects > 0)
        evaluation.add('denoSpu_{}_{}utts'.format(self._group_name, num_utterances), num_deno_corrects)
        evaluation.add('denoHitMass_{}_{}utts'.format(self._group_name, num_utterances), deno_hit_mass)

        # Sequence Evaluation
        first_seq_correct = None
        if example.logical_form:
            true_lf = _raw_lf(example.logical_form)
            seq_acc = (len(predictions) > 0 and true_lf == _raw_lf(predictions[0].decisions))
            evaluation.add('seqAcc_{}'.format(self._group_name), seq_acc)
            evaluation.add('seqAcc_{}_{}utts'.format(self._group_name, num_utterances), seq_acc)
            for i, prediction in enumerate(predictions):
                if true_lf == _raw_lf(prediction.decisions):
                    first_seq_correct = i
                    log.write('Seq correct at {}: {}\n\n'.format(i, prediction.decisions))
                    seq_hit = True
                    seq_hit_mass = probs[i]
                    break
            else:   # No prediction has a matching LF
                log.write('Seq correct not found.\n\n')
                seq_hit = False
                seq_hit_mass = 0.
            evaluation.add('seqHit_{}'.format(self._group_name), seq_hit)
            evaluation.add('seqHitMass_{}'.format(self._group_name), seq_hit_mass)
            evaluation.add('spuriousMass_{}'.format(self._group_name), deno_hit_mass - seq_hit_mass)
            evaluation.add('seqHit_{}_{}utts'.format(self._group_name, num_utterances), seq_hit)
            evaluation.add('seqHitMass_{}_{}utts'.format(self._group_name, num_utterances), seq_hit_mass)
            evaluation.add('spuriousMass_{}_{}utts'.format(self._group_name, num_utterances), deno_hit_mass - seq_hit_mass)
            # Separate by LF pattern
            pattern = _logical_form_pattern(example.logical_form)
            evaluation.add('denoAcc_{}_{}'.format(self._group_name, pattern), deno_acc)
            evaluation.add('denoHit_{}_{}'.format(self._group_name, pattern), num_deno_corrects > 0)
            evaluation.add('denoSpu_{}_{}'.format(self._group_name, pattern), num_deno_corrects)
            evaluation.add('seqAcc_{}_{}'.format(self._group_name, pattern), seq_acc)
            evaluation.add('seqHit_{}_{}'.format(self._group_name, pattern), seq_hit)
            evaluation.add('seqHitMass_{}_{}'.format(self._group_name, pattern), seq_hit_mass)
            evaluation.add('spuriousMass_{}_{}'.format(self._group_name, pattern), deno_hit_mass - seq_hit_mass)

        # Score breakdown: basic, attention, and soft_copy
        # First, gather all paths of interest
        paths_of_interest = [first_candidate, first_deno_correct, first_seq_correct]
        uniqued_paths_of_interest = list(set(x for x in paths_of_interest if x is not None))
        attentions, score_breakdowns = self._decoder.score_breakdown(
                [predictions[i] for i in uniqued_paths_of_interest])
        # Top candidate
        if first_candidate is None:
            log.write('[breakdown] Top candidate: NONE\n')
        else:
            log.write('[breakdown] Top candidate: {} / {}\n'.format(
                first_candidate, len(predictions)))
            self.log_score_breakdown(predictions[first_candidate],
                    attentions[uniqued_paths_of_interest.index(first_candidate)],
                    score_breakdowns[uniqued_paths_of_interest.index(first_candidate)], log)
        log.write('\n')
        # First deno correct
        if first_deno_correct is None:
            log.write('[breakdown] First deno correct: NONE\n')
        elif first_deno_correct == first_candidate:
            log.write('[breakdown] First deno correct: {} / {} (same as Top candidate)\n'.format(
                first_deno_correct, len(predictions)))
        else:
            log.write('[breakdown] First deno correct: {} / {}\n'.format(
                first_deno_correct, len(predictions)))
            self.log_score_breakdown(predictions[first_deno_correct],
                    attentions[uniqued_paths_of_interest.index(first_deno_correct)],
                    score_breakdowns[uniqued_paths_of_interest.index(first_deno_correct)], log)
        log.write('\n')
        # First seq correct
        if first_seq_correct is None:
            log.write('[breakdown] First seq correct: NONE\n')
        elif first_seq_correct == first_candidate:
            log.write('[breakdown] First seq correct: {} / {} (same as Top candidate)\n'.format(
                first_seq_correct, len(predictions)))
        elif first_seq_correct == first_deno_correct:
            log.write('[breakdown] First seq correct: {} / {} (same as First deno correct)\n'.format(
                first_seq_correct, len(predictions)))
        else:
            log.write('[breakdown] First seq correct: {} / {}\n'.format(
                first_seq_correct, len(predictions)))
            self.log_score_breakdown(predictions[first_seq_correct],
                    attentions[uniqued_paths_of_interest.index(first_seq_correct)],
                    score_breakdowns[uniqued_paths_of_interest.index(first_seq_correct)], log)
        log.write('\n')

        # Print the Beams
        if log_all_beams:
            if not example.logical_form:
                match_gold_prefix = lambda x: False
            else:
                match_gold_prefix = lambda x: true_lf[:len(x)] == x
            for step, beam in enumerate(itertools.chain(intermediate_beams, [final_beam])):
                log.write('Beam at step {}:\n'.format(step))
                match_any = False
                for path in beam:
                    match = match_gold_prefix(_raw_lf(path.decisions))
                    match_any = match_any or match
                    log.write('{match} {decisions} ({score}) -- {embed}\n'.format(
                        match='@' if match else ' ', decisions=path.decisions, score=ranker(path),
                        embed=path[-1].pretty_embed if len(path) > 0 else None
                    ))
                if example.logical_form:
                    evaluation.add('seqOra_{}_{}'.format(self._group_name, step), match_any)
            log.write('\n')

    def log_score_breakdown(self, path, attention, score_breakdown, log):
        decisions = path.decisions
        log.write('Logical form: {}\n'.format(' '.join(str(x) for x in decisions)))
        for i, case in enumerate(path):
            log.write('  Step {}: {} ___\n'.format(i, ' '.join(str(x) for x in decisions[:i])))

            utterance = case.current_utterance
            capped_utterance = utterance[:min(len(utterance), len(attention[i]))]

            # Attention
            log.write('  {}\n'.format(' '.join('{:>6}'.format(x.encode('utf8')[:6])
                for x in capped_utterance)))
            log.write('  {}\n'.format(' '.join('{:6.3f}'.format(x)
                for x in attention[i][:len(capped_utterance)])))
            attention_rank = sorted(list(range(len(capped_utterance))),
                    key=lambda j: -attention[i][j])
            log.write('  {}\n'.format(' '.join('{:>6}'.format(
                '*' if j in attention_rank[:3] else '')
                for j in range(len(capped_utterance)))))
            # Sort by total logit
            choice_indices = sorted(list(range(len(case.choices))),
                key=lambda j: -sum(score_breakdown[i][j]))
            for j in choice_indices:
                is_chosen = (case.choices[j] == decisions[i])
                log.write('  {:>15} {} | {} | {:7.3f}\n'.format(
                    _abbrev_predicate(case.choices[j])[:15],
                    '@' if is_chosen else ' ',
                    ' '.join('{:7.3f}'.format(x) for x in
                        score_breakdown[i][j]),
                    sum(score_breakdown[i][j])))


# Helper function
def _raw_lf(lf):
    """Return the logical form without EOU"""
    return [x.name for x in lf if x.name != EOU]


def _logical_form_pattern(lf):
    lf = ' '.join(_raw_lf(lf))
    if re.match(r'^type-row count$', lf):
        return 'row_count'
    if re.match(r'^fb:cell\.\w+ fb:row\.row\.\w+ count$', lf):
        return 'ent_count'
    if re.match(r'^fb:cell\.\w+ fb:row\.row\.\w+ !fb:row\.row\.\w+$', lf):
        return 'lookup'
    if re.match(r'^fb:cell\.\w+ fb:row\.row\.\w+ !?fb:row\.row\.next !fb:row\.row\.\w+$', lf):
        return 'next_prev'
    if re.match(r'^type-row x !fb:row\.row\.index arg(min|max) !fb:row\.row\.\w+$', lf):
        return 'first_last'
    if re.match(r'^type-row !fb:row\.row\.index (min|max) fb:row\.row\.index !fb:row\.row\.\w+$', lf):
        return 'first_last'
    if re.match(r'^type-row !fb:row\.row\.\w+ x fb:row\.row\.\w+ count argmax$', lf):
        return 'most_freq'
    return 'unknown'


def _abbrev_predicate(x):
    x = str(x)
    if x.startswith('fb:row.row.'):
        return 'r.' + x[11:]
    if x.startswith('!fb:row.row.'):
        return '!r.' + x[12:]
    if x.startswith('fb:cell.cell.'):
        return 'n.' + x[13:]
    if x.startswith('!fb:cell.cell.'):
        return '!n.' + x[14:]
    if x.startswith('fb:cell.'):
        return 'c.' + x[8:]
    if x.startswith('fb:part.'):
        return 'p.' + x[8:]
    return x
