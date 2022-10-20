import os
import random


import numpy as np
import tensorflow as tf

import gtd.ml.experiment
from dependency.data_directory import DataDirectory
from gtd.chrono import verboserate
from gtd.ml.model import TokenEmbedder
from gtd.ml.utils import guarantee_initialized_variables
from gtd.utils import cached_property, as_batches, random_seed, sample_if_large
from strongsup.decoder import Decoder
from strongsup.domain import get_domain
from strongsup.embeddings import (
        StaticPredicateEmbeddings, GloveEmbeddings, TypeEmbeddings,
        RLongPrimitiveEmbeddings)
from strongsup.evaluation import Evaluation, BernoulliSequenceStat
from strongsup.example import Example, DelexicalizedContext
from strongsup.parse_case import ParsePath
from strongsup.parse_model import (
    UtteranceEmbedder,
    CombinedPredicateEmbedder, DynamicPredicateEmbedder,
    PositionalPredicateEmbedder, DelexicalizedDynamicPredicateEmbedder,
    HistoryEmbedder,
    SimplePredicateScorer, AttentionPredicateScorer,
    SoftCopyPredicateScorer, PredicateScorer,
    CrossEntropyLossModel, LogitLossModel,
    ParseModel, TrainParseModel,
    ExecutionStackEmbedder, RLongObjectEmbedder)
from strongsup.utils import OptimizerOptions
from strongsup.value_function import ValueFunctionExample
from strongsup.visualizer import Visualizer


class Experiments(gtd.ml.experiment.Experiments):
    def __init__(self, check_commit=True):
        """Create Experiments.

        If check_commit is true, this will not allow you to run old experiments
        without being on the correct commit number, or old experiments where
        the working directory was not clean.
        """
        data_dir = DataDirectory.experiments
        src_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        default_config = os.path.join(src_dir, 'configs', 'debug.txt')
        super(Experiments, self).__init__(data_dir, src_dir, Experiment, default_config, check_commit=check_commit)


class Experiment(gtd.ml.experiment.TFExperiment):
    """Encapsulates the elements of a training run."""

    def __init__(self, config, save_dir):
        super(Experiment, self).__init__(config, save_dir)
        self.workspace.add_file('train_visualize', 'train_visualizer.txt')
        self.workspace.add_file('valid_visualize', 'valid_visualizer.txt')
        self.workspace.add_file('full_eval', 'full_eval_at_{step}.txt')
        self.workspace.add_file('codalab', 'codalab.json')
        self._domain = get_domain(config)

        self._train_parse_model = self._build_train_parse_model()
        self._decoder = self._build_decoder(self.train_parse_model)

        self._train_visualizer = Visualizer(self.decoder, self.workspace.train_visualize,
                                'train', train=True)

        self._valid_visualizer = Visualizer(self.decoder, self.workspace.valid_visualize,
                                'valid', train=False)

        # Reload weights if they exist. Otherwise, initialize weights.
        try:
            self.saver.restore()
            print('Successfully reloaded the weights')
        except IOError:
            # NOTE: use this instead of tf.initialize_all_variables()!
            # That op will overwrite Keras initializations.
            sess = tf.get_default_session()
            guarantee_initialized_variables(sess)
            print('Weights initialized')

    @property
    def train_parse_model(self):
        return self._train_parse_model

    @property
    def parse_model(self):
        return self.train_parse_model.parse_model

    @property
    def decoder(self):
        return self._decoder

    @property
    def path_checker(self):
        return self._domain.path_checker

    @property
    def train_visualizer(self):
        return self._train_visualizer

    @property
    def valid_visualizer(self):
        return self._valid_visualizer

    def _build_train_parse_model(self):
        """Construct the TrainParseModel.

        If weights have been saved to disk, restore those weights.

        Returns:
            TrainParseModel
        """
        config = self.config.parse_model
        delexicalized = self.config.delexicalized

        # Glove embeddings have embed_dim 100
        glove_embeddings = GloveEmbeddings(vocab_size=20000)
        type_embeddings = TypeEmbeddings(embed_dim=50, all_types=self._domain.all_types)

        # set up word embeddings
        word_embedder = TokenEmbedder(glove_embeddings, 'word_embeds', trainable=config.train_word_embeddings)
        type_embedder = TokenEmbedder(type_embeddings, 'type_embeds')

        # build utterance embedder
        utterance_embedder = UtteranceEmbedder(word_embedder, lstm_dim=config.utterance_embedder.lstm_dim,
                                               utterance_length=config.utterance_embedder.utterance_length)

        # build predicate embedder

        # dynamic
        if delexicalized:
            dyn_pred_embedder = DelexicalizedDynamicPredicateEmbedder(utterance_embedder.hidden_states_by_utterance,
                                                                      type_embedder)
        else:
            dyn_pred_embedder = DynamicPredicateEmbedder(word_embedder, type_embedder)
            if config.predicate_positions:
                dyn_pred_embedder = PositionalPredicateEmbedder(dyn_pred_embedder)

        # static
        static_pred_embeddings = StaticPredicateEmbeddings(
                dyn_pred_embedder.embed_dim,   # matching dim
                self._domain.fixed_predicates)
        static_pred_embedder = TokenEmbedder(static_pred_embeddings, 'static_pred_embeds')

        # combined
        pred_embedder = CombinedPredicateEmbedder(static_pred_embedder, dyn_pred_embedder)

        # build history embedder
        if config.condition_on_history:
            history_embedder = HistoryEmbedder(pred_embedder, config.history_length)
        else:
            history_embedder = None

        # build execution stack embedder
        if config.condition_on_stack:
            max_stack_size = self.config.decoder.prune.max_stack_size
            max_list_size = config.stack_embedder.max_list_size
            primitive_dim = config.stack_embedder.primitive_dim
            object_dim = config.stack_embedder.object_dim

            primitive_embeddings = RLongPrimitiveEmbeddings(primitive_dim)
            stack_primitive_embedder = TokenEmbedder(primitive_embeddings, 'primitive_embeds', trainable=True)

            # TODO(kelvin): pull this out as its own method
            assert self.config.dataset.domain == 'rlong'
            sub_domain = self.config.dataset.name
            attrib_extractors = [lambda obj: obj.position]
            if sub_domain == 'scene':
                attrib_extractors.append(lambda obj: obj.shirt)
                attrib_extractors.append(lambda obj: obj.hat)
                pass
            elif sub_domain == 'alchemy':
                # skipping chemicals attribute for now, because it is actually a list
                attrib_extractors.append(lambda obj: obj.color if obj.color is not None else 'color-na')
                attrib_extractors.append(lambda obj: obj.amount)
            elif sub_domain == 'tangrams':
                attrib_extractors.append(lambda obj: obj.shape)
            elif sub_domain == 'undograms':
                attrib_extractors.append(lambda obj: obj.shape)
            else:
                raise ValueError('No stack embedder available for sub-domain: {}.'.format(sub_domain))

            stack_object_embedder = RLongObjectEmbedder(attrib_extractors, stack_primitive_embedder,
                                                        max_stack_size, max_list_size)

            stack_embedder = ExecutionStackEmbedder(stack_primitive_embedder, stack_object_embedder,
                                                    max_stack_size=max_stack_size,  max_list_size=max_list_size,
                                                    project_object_embeds=True, abstract_objects=False)
        else:
            stack_embedder = None

        def scorer_factory(query_tensor):
            simple_scorer = SimplePredicateScorer(query_tensor, pred_embedder)
            attention_scorer = AttentionPredicateScorer(query_tensor, pred_embedder, utterance_embedder)
            soft_copy_scorer = SoftCopyPredicateScorer(attention_scorer.attention_on_utterance.logits,
                                                       disable=not config.soft_copy
                                                       )  # note that if config.soft_copy is None, then soft_copy is disabled
            scorer = PredicateScorer(simple_scorer, attention_scorer, soft_copy_scorer)
            return scorer

        parse_model = ParseModel(pred_embedder, history_embedder, stack_embedder,
                utterance_embedder, scorer_factory, config.h_dims,
                self._domain, delexicalized)

        if self.config.decoder.normalization == 'local':
            loss_model_factory = CrossEntropyLossModel
        else:
            loss_model_factory = LogitLossModel
        train_parse_model = TrainParseModel(parse_model, loss_model_factory,
                self.config.learning_rate,
                OptimizerOptions(self.config.optimizer),
                self.config.get('train_batch_size'))

        return train_parse_model

    def _build_decoder(self, train_parse_model):
        return Decoder(train_parse_model, self.config.decoder, self._domain)

    @cached_property
    def _examples(self):
        train, valid, final = self._domain.load_datasets()

        def delexicalize_examples(examples):
            delex_examples = []
            for ex in examples:
                delex_context = DelexicalizedContext(ex.context)
                delex_ex = Example(delex_context, answer=ex.answer, logical_form=ex.logical_form)
                delex_examples.append(delex_ex)
            return delex_examples

        if self.config.delexicalized:
            train = delexicalize_examples(train)
            valid = delexicalize_examples(valid)
            final = delexicalize_examples(final)
        return train, valid, final

    @property
    def train_examples(self):
        return self._examples[0]

    @property
    def valid_examples(self):
        return self._examples[1]

    @property
    def final_examples(self):
        return self._examples[2]

    def train(self):
        decoder = self.decoder
        eval_steps = self.config.timing.eval
        big_eval_steps = self.config.timing.big_eval
        save_steps = self.config.timing.save
        self.evaluate(step=decoder.step)  # evaluate once before training begins

        while True:
            train_examples = random.sample(self.train_examples, k=len(self.train_examples))  # random shuffle
            train_examples = verboserate(train_examples, desc='Streaming training Examples')
            for example_batch in as_batches(train_examples, self.config.batch_size):
                decoder.train_step(example_batch)
                step = decoder.step

                self.report_cache_stats(step)
                if (step + 1) % save_steps == 0:
                    self.saver.save(step)
                if (step + 1) % eval_steps == 0:
                    self.evaluate(step)
                if (step + 1) % big_eval_steps == 0:
                    self.big_evaluate(step)
                if step >= self.config.max_iters:
                    self.evaluate(step)
                    self.saver.save(step)
                    return

    # def supervised_train(self):
    #     train_parse_model = self.train_parse_model
    #     eval_time = Pulse(self.config.timing.eval)
    #     supervised_eval_time = Pulse(self.config.timing.supervised_eval)
    #     cases = examples_to_supervised_cases(self.train_examples)
    #     while True:
    #         for case_batch in as_batches(cases, self.config.batch_size):
    #             weights = [1.0 / len(case_batch)] * len(case_batch)
    #             train_parse_model.train_step(case_batch, weights, self.config.decoder.inputs_caching)
    #             step = train_parse_model.step
    #             self.report_cache_stats(step)
    #             self.saver.interval_save(step, self.config.timing.save)
    #             if eval_time():
    #                 self.evaluate(step)
    #                 eval_time.reset()
    #             if supervised_eval_time():
    #                 self.supervised_evaluate(step)
    #                 supervised_eval_time.reset()
    #             if step >= self.config.max_iters:
    #                 self.evaluate(step)
    #                 self.saver.save(step)
    #                 return

    def report_cache_stats(self, step):
        return              # Don't log these
        parse_model = self.parse_model
        scorer_cache = parse_model._scorer.inputs_to_feed_dict_cached
        pred_cache = parse_model._pred_embedder.inputs_to_feed_dict_cached
        self.tb_logger.log('cache_scorer_size', scorer_cache.cache_size, step)
        self.tb_logger.log('cache_predEmbedder_size', pred_cache.cache_size, step)
        self.tb_logger.log('cache_scorer_hitRate', scorer_cache.hit_rate, step)
        self.tb_logger.log('cache_predEmbedder_hitRate', pred_cache.hit_rate, step)

    def supervised_evaluate(self, step):
        train_parse_model = self.train_parse_model

        def case_sample(examples):
            """Get a random sample of supervised ParseCases."""
            with random_seed(0):
                example_sample = sample_if_large(examples, 30)
            return list(examples_to_supervised_cases(example_sample))

        def report_loss(cases, name):
            weights = [1.0] * len(cases)
            loss = train_parse_model.compute(train_parse_model.loss, cases, weights,
                    caching=self.config.decoder.inputs_caching)
            self.tb_logger.log(name, loss, step)

        report_loss(case_sample(self.train_examples), 'loss_train')
        report_loss(case_sample(self.valid_examples), 'loss_val')

    def evaluate(self, step):
        print('Evaluate at step {}'.format(step))
        num_examples = self.config.num_evaluate_examples
        with random_seed(0):
            train_sample = sample_if_large(self.train_examples, num_examples,
                    replace=False)
        with random_seed(0):
            valid_sample = sample_if_large(self.valid_examples, num_examples,
                    replace=False)
        train_eval = self.evaluate_on_examples(step, train_sample, self.train_visualizer)
        valid_eval = self.evaluate_on_examples(step, valid_sample, self.valid_visualizer)

        # Log to TensorBoard
        train_eval.json_summarize(self.workspace.codalab, step)
        train_eval.tboard_summarize(self.tb_logger, step)
        valid_eval.json_summarize(self.workspace.codalab, step)
        valid_eval.tboard_summarize(self.tb_logger, step)

    def evaluate_on_examples(self, step, examples, visualizer):
        evaluation = Evaluation()
        examples = verboserate(examples, desc='Decoding {} examples'.format(visualizer.group_name))
        visualizer.reset(step=step)
        for ex_batch in as_batches(examples, self.config.batch_size):
            beams, batch_evaluation = visualizer.predictions(ex_batch)
            evaluation.add_evaluation(batch_evaluation)

            # collect value function examples
            value_function = self.decoder._value_function
            vf_examples = []
            for example, beam in zip(ex_batch, beams):
                vf_examples.extend(ValueFunctionExample.examples_from_paths(beam, example))

            # compute ValueFunction metrics
            vf_loss = value_function.loss(vf_examples)
            predicted_values = value_function.values([ex.case for ex in vf_examples])
            avg_predicted_value = np.mean(predicted_values)
            evaluation.add('valueFunctionLoss', vf_loss)
            evaluation.add('avgPredictedValue', avg_predicted_value)

        return evaluation

    def big_evaluate(self, step, num_samples=None):
        """Run more comprehensive evaluation of the model.

        How this differs from `self.evaluate`:
        - Compute confidence intervals for denotational accuracy estimates
        - Option to evaluate on a custom/larger number of samples
        - Due to the larger number of samples, don't print everything out to a visualizer.
        - Save stats to a file.

        Args:
            step (int)
            num_samples (# samples to evaluate on, for both train and test)
        """
        if num_samples is None:
            num_samples = self._config.num_evaluate_examples_big
        full_eval_path = self.workspace.full_eval.format(step=step)
        silent_visualizer = Visualizer(self.decoder, '/dev/null', 'silent', train=False)

        def evaluate_helper(examples, prefix):
            with random_seed(0):
                sample = sample_if_large(examples, num_samples, replace=False)
            eval = self.evaluate_on_examples(step=step, examples=sample, visualizer=silent_visualizer)

            # wrap with BernoulliSequenceStat, for conf intervals
            for name, stat in list(eval.stats.items()):
                if name.startswith('denoAcc'):
                    eval.stats[name] = BernoulliSequenceStat(stat)

            with open(full_eval_path, 'a') as f:
                eval.summarize(f, prefix=prefix)

            eval.tboard_summarize(self.tb_logger, step, prefix=prefix)
            eval.json_summarize(self.workspace.codalab, step, prefix=prefix)

        evaluate_helper(self.final_examples, 'FINAL')
        evaluate_helper(self.valid_examples, 'VALID')


def example_to_supervised_cases(example):
    """Convert Example to a list of supervised ParseCases.

    Only possible if example.logical_form is known.

    Args:
        example (Example)

    Returns:
        list[ParseCase]
    """
    path = ParsePath([], context=example.context)
    predicates = example.logical_form
    cases = []
    for pred in predicates:
        case = path.extend()
        if pred not in case.choices:
            case.choices.append(pred)
        case.decision = pred
        cases.append(case)
        path = case.path
    return cases


def examples_to_supervised_cases(examples):
    """Return a Generator of supervised ParseCases."""
    for example in verboserate(examples, desc='Streaming supervised ParseCases'):
        for case in example_to_supervised_cases(example):
            yield case
