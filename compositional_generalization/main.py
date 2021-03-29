import argparse
import os
import random
import time
import unicodedata
from functools import partial
import torch
from torch import nn
from tqdm import tqdm

from model import HRLModel, PAD_token, EOS_token
from utils import AverageMeter
from utils import VisualizeLogger
from utils import get_logger
import numpy as np

USE_CUDA = torch.cuda.is_available()
global_step = 0


class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"x1": 3, "x2": 4, "x3": 5, "x4": 6}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "x1", 4: "x2", 5: "x3", 6: "x4"}
        self.n_words = 7  # Count default tokens

    def vocab_size(self):
        return len(self.word2index.keys())

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

        for word in keep_words:
            self.index_word(word)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    return s


def read_data(lang1, lang2, task_name):
    print("Reading dataset from task {}...".format(task_name))
    lines_train = open('./data/tasks_train_{}.txt'.format(task_name), encoding='utf-8'). \
        read().strip().split('\n')
    lines_test = open('./data/tasks_test_{}.txt'.format(task_name), encoding='utf-8'). \
        read().strip().split('\n')

    pairs_train = [[normalize_string(s) for s in l.lstrip('IN: ').split(' OUT: ')] for l in lines_train]
    pairs_test = [[normalize_string(s) for s in l.lstrip('IN: ').split(' OUT: ')] for l in lines_test]

    _input_lang = Lang(lang1)
    _output_lang = Lang(lang2)

    return _input_lang, _output_lang, pairs_train, pairs_test


def prepare_dataset(lang1, lang2, task_name):
    global input_lang
    global output_lang
    input_lang, output_lang, pairs_train, pairs_test = read_data(lang1, lang2, task_name)
    for pair in pairs_train:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    if task_name == "addjump":
        # remove duplicated JUMP command
        pairs_train = list(set([tuple(item) for item in pairs_train]))
        pairs_train = [list(item) for item in pairs_train]

    return input_lang, output_lang, pairs_train, pairs_test


def get_bound_idx(pairs, length):
    index = 0
    for i, pair in enumerate(pairs):
        if len(pair[0].split()) <= length:
            index = i
        else:
            return index + 1


def random_batch(pair):
    input_seqs = []
    target_seqs = []

    input_seqs.append(indexes_from_sentence(input_lang, pair[0], 'input'))
    target_seqs.append(indexes_from_sentence(output_lang, pair[1], 'output'))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_mask = torch.zeros((len(input_lengths), max(input_lengths)), dtype=torch.float32)
    for idx, length in enumerate(input_lengths):
        input_mask[idx, :length] = 1
    target_mask = torch.zeros((len(target_lengths), max(target_lengths)), dtype=torch.float32)
    for idx, length in enumerate(target_lengths):
        target_mask[idx, :length] = 1

    input_var = torch.LongTensor(input_padded)
    target_var = torch.LongTensor(target_padded)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, input_mask, target_var, target_lengths


def indexes_from_sentence(lang, sentence, type):
    if type == 'input':
        return [lang.word2index[word] for word in sentence.split(' ')]
    if type == 'output':
        return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def make_path_preparations(args, run_mode):
    seed = args.random_seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if run_mode == 'train':
        log_dir = os.path.split(args.logs_path)[0]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        _logger = get_logger(f"{args.logs_path}.log")
        print(f"{args.logs_path}.log")
        _logger.info(f"random seed: {seed}")

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        _logger.info(f"checkpoint's dir is: {args.model_dir}")
        _visualizer = VisualizeLogger(summary_dir=args.model_dir)
    else:
        _logger = None
        _visualizer = None

    return _logger, _visualizer


def prepare_optimisers(args, logger, policy_parameters, environment_parameters):
    if args.env_optimizer == "adam":
        env_opt_class = torch.optim.Adam
    elif args.env_optimizer == "amsgrad":
        env_opt_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.env_optimizer == "adadelta":
        env_opt_class = torch.optim.Adadelta
    else:
        env_opt_class = torch.optim.SGD

    if args.pol_optimizer == "adam":
        pol_opt_class = torch.optim.Adam
    elif args.pol_optimizer == "amsgrad":
        pol_opt_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.pol_optimizer == "adadelta":
        pol_opt_class = torch.optim.Adadelta
    else:
        pol_opt_class = torch.optim.SGD

    optimizer = {"policy": pol_opt_class(params=policy_parameters, lr=args.pol_lr, weight_decay=args.l2_weight),
                 "env": env_opt_class(params=environment_parameters, lr=args.env_lr, weight_decay=args.l2_weight)}
    return optimizer


def perform_env_optimizer_step(optimizer, model, args):
    if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.get_environment_parameters(),
                                 max_norm=args.clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer["env"].step()
    optimizer["env"].zero_grad()


def perform_policy_optimizer_step(optimizer, model, args):
    if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.get_policy_parameters(),
                                 max_norm=args.clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer["policy"].step()
    optimizer["policy"].zero_grad()


def visualize_tree(seq, tree_actions_batch, sr_actions_batch, swr_actions_batch):
    seq_list = seq.split()
    assert len(seq_list) == len(swr_actions_batch)
    for idx, swr_action in enumerate(swr_actions_batch):
        if swr_action[0, 0] == 1:
            seq_list[idx] = "[" + seq_list[idx] + "]"

    for tree_action_batch, sr_action_batch in zip(tree_actions_batch, sr_actions_batch):

        if tree_action_batch is None:
            break
        tree_action = tree_action_batch[0]
        sr_action = sr_action_batch[0]
        merge_idx = tree_action.tolist().index(1)
        sr_idx = sr_action.tolist().index(1)
        if sr_idx == 1:
            seq_list = seq_list[:merge_idx] + ['(' + ' '.join(seq_list[merge_idx:merge_idx + 2]) + ')'] + seq_list[
                                                                                                          merge_idx + 2:]
        else:
            seq_list = seq_list[:merge_idx] + ['[' + ' '.join(seq_list[merge_idx:merge_idx + 2]) + ']'] + seq_list[
                                                                                                          merge_idx + 2:]

    return seq_list[0]


def evaluate(test_data, model, device):
    loading_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()
    model.eval()
    start = time.time()

    debug_info = {}
    with torch.no_grad():
        progress_bar = tqdm(range(len(test_data)))
        for idx in progress_bar:
            test_data_example = test_data[idx]
            tokens, tokens_length, mask, labels, labels_length = random_batch(test_data_example)
            tokens = tokens.to(device=device)
            mask = mask.to(device=device)
            loading_time_meter.update(time.time() - start)

            pred_labels, tree_sr_log_prob, tree_sr_rewards, decoder_log_probs, decode_rewards, tree_actions, sr_actions, swr_actions, normalized_entropy = \
                model(test_data_example, tokens, mask, debug_info=debug_info)

            normalized_entropy = normalized_entropy.mean()
            accuracy = [1. if (pred_labels == test_data_example[1]) else 0.]
            accuracy = torch.tensor(accuracy).mean()
            ce_loss = accuracy
            n = mask.shape[0]
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)
            progress_bar.set_description("Test Acc {:.1f}%".format(accuracy_meter.avg * 100))

    return accuracy_meter.avg


def validate(valid_data, model, epoch, device, logger):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()

    if len(valid_data) > 1000:
        # to accelerate
        valid_data = [random.choice(valid_data) for _ in range(1000)]

    visualizer.update_validate_size(len(valid_data))

    model.eval()
    start = time.time()

    debug_info = {}
    with torch.no_grad():
        for idx, valid_data_example in enumerate(valid_data):
            tokens, tokens_length, mask, labels, labels_length = random_batch(valid_data_example)

            tokens = tokens.to(device=device)
            mask = mask.to(device=device)
            loading_time_meter.update(time.time() - start)

            pred_labels, tree_sr_log_prob, tree_sr_rewards, decoder_log_probs, decode_rewards, tree_actions, sr_actions, swr_actions, normalized_entropy = \
                model(valid_data_example, tokens, mask, debug_info=debug_info)

            """
            logging into visualizer
            """
            debug_info['tree_sr_rewards'] = tree_sr_rewards
            debug_info['decode_rewards'] = decode_rewards
            seq = " ".join([input_lang.index2word[token.data.item()] for token in tokens[0]])
            tree = visualize_tree(seq, tree_actions, sr_actions, swr_actions)
            visualizer.log_text(valid_data_example[1], tree, pred_labels, seq, debug_info)
            visualizer.update_step()

            normalized_entropy = normalized_entropy.mean()
            accuracy = [1. if (pred_labels == valid_data_example[1]) else 0.]
            accuracy = torch.tensor(accuracy).mean()

            ce_loss = accuracy
            n = mask.shape[0]
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)
            batch_time_meter.update(time.time() - start)
            start = time.time()

    visualizer.log_performance(accuracy_meter.avg)
    visualizer.update_epoch()

    logger.info(f"Valid: epoch: {epoch} ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"n_entropy: {n_entropy_meter.avg:.4f} "
                f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f}")
    model.train()
    return accuracy_meter.avg


def train(train_data, valid_data, model, optimizer, epoch, args, logger,
          total_batch_num, data_len, regular_weight):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()
    prob_ratio_meter = AverageMeter()
    reward_std_meter = AverageMeter()

    device = args.gpu_id
    model.train()
    start = time.time()

    # simple data augmentation for lasting longer epochs for MiniSCAN
    if len(train_data) < 100:
        train_data = [pair for pair in train_data for _ in range(8)]
    elif len(train_data) < 500:
        train_data = [pair for pair in train_data for _ in range(2)]

    random.shuffle(train_data)
    batch_size = args.accumulate_batch_size

    if len(train_data) % batch_size == 0:
        batch_num = len(train_data) // batch_size
    else:
        batch_num = len(train_data) // batch_size + 1

    val_accuracy = 0.
    for batch_idx in range(batch_num):

        if (batch_idx + 1) * batch_size < len(train_data):
            train_pairs = train_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        else:
            train_pairs = train_data[batch_idx * batch_size:]
            batch_size = len(train_pairs)

        total_batch_num += batch_size

        loading_time_meter.update(time.time() - start)

        normalized_entropy_samples = []
        ts_log_prob_samples = []
        decode_log_prob_samples = []
        ts_rewards_samples = []
        decode_rewards_samples = []
        rewards_all = []
        root_rewards_all = []
        accuracy_samples = []

        sample_num = 10
        for example_idx in range(batch_size):
            for sample_idx in range(sample_num):
                train_pair = train_pairs[example_idx]
                tokens, tokens_length, mask, labels, labels_length = random_batch(train_pair)
                tokens = tokens.to(device=device)
                mask = mask.to(device=device)

                pred_labels, tree_sr_log_prob, tree_sr_rewards, decoder_log_probs, decode_rewards, tree_actions, sr_actions, swr_actions, normalized_entropy = \
                    model(train_pair, tokens, mask, is_test=False, epoch=epoch)

                accuracy = 1. if (pred_labels == train_pair[1]) else 0.

                normalized_entropy_samples.append(normalized_entropy)

                ts_log_prob_samples.append(tree_sr_log_prob)
                ts_rewards_samples.append(tree_sr_rewards)

                decode_log_prob_samples.append(decoder_log_probs)
                decode_rewards_samples.append(decode_rewards)

                rewards_all = rewards_all + decode_rewards
                accuracy_samples.append(accuracy)

                root_rewards_all.append(decode_rewards[-1])

        normalized_entropy_samples = torch.cat(normalized_entropy_samples, dim=0)
        accuracy_samples = torch.tensor(accuracy_samples).cuda()
        rewards_all = torch.tensor(rewards_all).cuda()

        baseline = rewards_all.mean()
        accuracy = accuracy_samples.mean()

        loss_all = []

        for idy, ts_rewards in enumerate(ts_rewards_samples):
            ts_actions_log_prob = torch.cat(ts_log_prob_samples[idy], dim=0)

            ts_rewards = torch.tensor(ts_rewards).cuda()

            if baseline:
                ts_rewards = ts_rewards - baseline

            ts_prob_ratio = (ts_actions_log_prob - ts_actions_log_prob.detach()).exp()
            ts_loss = (ts_prob_ratio * ts_rewards).mean().unsqueeze(0)

            decode_rewards = decode_rewards_samples[idy]
            decode_actions_log_prob = torch.cat(decode_log_prob_samples[idy], dim=0)
            decode_rewards = torch.tensor(decode_rewards).cuda()

            if baseline:
                decode_rewards = decode_rewards - baseline

            decode_prob_ratio = (decode_actions_log_prob - decode_actions_log_prob.detach()).exp()
            decode_loss = (decode_prob_ratio * decode_rewards).mean().unsqueeze(0)

            loss_all.append(ts_loss + decode_loss)

        loss_avg = torch.cat(loss_all, dim=0).mean()

        loss = loss_avg - regular_weight * normalized_entropy_samples.mean()

        loss.backward()
        perform_policy_optimizer_step(optimizer, model, args)
        perform_env_optimizer_step(optimizer, model, args)

        normalized_entropy = normalized_entropy.mean()
        n = mask.shape[0]

        ce_loss = rewards_all.mean()
        accuracy_meter.update(accuracy.item(), n)
        ce_loss_meter.update(ce_loss.item(), n)
        reward_std_meter.update(rewards_all.std().item(), n)

        n_entropy_meter.update(normalized_entropy.item(), n)
        prob_ratio_meter.update((1.0 - loss_avg.detach()).abs().mean().item(), n)
        batch_time_meter.update(time.time() - start)

        global global_step
        global_step += 1

        if batch_num <= 500:
            val_num = batch_num
        else:
            val_num = 250

        if (batch_idx + 1) % (val_num) == 0:
            logger.info(f"Train: epoch: {epoch} batch_idx: {batch_idx + 1} ce_loss: {ce_loss_meter.avg:.4f} "
                        f"reward_std: {reward_std_meter.avg:.4f} "
                        f"n_entropy: {n_entropy_meter.avg:.4f} loading_time: {loading_time_meter.avg:.4f} "
                        f"batch_time: {batch_time_meter.avg:.4f}")
            logger.info(f"total_batch_num: {total_batch_num} cir: {data_len}")

            val_accuracy = validate(valid_data, model, epoch, device, logger)

            global best_model_path
            logger.info("saving model...")
            best_model_path = f"{args.model_dir}/{epoch}-{batch_idx}.mdl"
            torch.save({"epoch": epoch, "batch_idx": batch_idx, "state_dict": model.state_dict()}, best_model_path)
            model.train()

        start = time.time()

        if val_accuracy >= 0.99:
            break

    return val_accuracy, total_batch_num


def train_model(args, task_name, logger):
    global input_lang
    global output_lang

    input_lang, output_lang, pairs_train, _ = prepare_dataset('nl', 'action', task_name)

    index = [i for i in range(len(pairs_train))]
    random.shuffle(index)
    train_size = int(0.8 * len(pairs_train))
    dev_size = len(pairs_train) - train_size
    train_idxs, dev_idxs = torch.utils.data.random_split(index, [train_size, dev_size])

    train_pairs_all = [pairs_train[idx] for idx in train_idxs]
    dev_pairs_all = [pairs_train[idx] for idx in dev_idxs]
    for pair in dev_pairs_all:
        if len(pair[0].split()) <= 4:
            train_pairs_all.append(pair)

    train_data, dev_data = train_pairs_all, dev_pairs_all

    train_data.sort(key=lambda p: len(p[0].split()))
    maximum_lesson = len(train_data[-1][0].split())

    dev_data = list(set([tuple(item) for item in dev_data]))
    dev_data.sort(key=lambda p: len(p[0].split()))
    dev_data = [list(item) for item in dev_data]

    print(random.choice(train_pairs_all))
    print(random.choice(dev_pairs_all))

    args.vocab_size = input_lang.n_words
    args.label_size = output_lang.n_words

    model = HRLModel(x_ratio_rate=args.simplicity_ratio,
                     encode_mode=args.encode_mode,
                     decay_r=args.decay_r,
                     vocab_size=args.vocab_size,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     label_dim=args.label_size,
                     composer_leaf=args.composer_leaf,
                     composer_trans_hidden=args.composer_trans_hidden,
                     input_lang=input_lang,
                     output_lang=output_lang).cuda(args.gpu_id)

    optimizer = prepare_optimisers(args, logger,
                                   policy_parameters=model.get_policy_parameters(),
                                   environment_parameters=model.get_environment_parameters())

    data_len = 3
    epoch_count = 0
    # default is 1 lesson
    cir_epoch_dict = {
        3: 30,
        4: 30,
        5: 20,
        6: 10,
        7: 5
    }

    regular_weight = args.init_regular_weight
    print('Start lesson ', data_len)
    total_batch_num = 0
    for epoch in range(args.max_epoch):

        if data_len in cir_epoch_dict:
            # training epochs
            cir_epoch_num = cir_epoch_dict[data_len]
        else:
            cir_epoch_num = 1

        train_lesson_idx = get_bound_idx(train_data, data_len)
        dev_lesson_idx = get_bound_idx(dev_data, data_len)
        val_accuracy, total_batch_num = train(train_data[:train_lesson_idx],
                                              dev_data[:dev_lesson_idx], model, optimizer,
                                              epoch, args, logger,
                                              total_batch_num, data_len, regular_weight)

        if data_len == maximum_lesson and val_accuracy >= 0.99:
            print("Finish Training. Training Succeed :)")
            break

        epoch_count += 1
        if epoch_count == cir_epoch_num or val_accuracy >= 0.99:
            # validate on all dev data
            if val_accuracy >= 0.99:
                val_accuracy_all = validate(dev_data, model, epoch, args.gpu_id, logger)
                if val_accuracy_all >= 0.99:
                    print("Early Stopped. Training Succeed :)")
                    break
            if data_len < maximum_lesson:
                print('Lesson ', data_len, 'completed at', epoch)
                data_len += 1
                regular_weight = max(args.regular_decay_rate * regular_weight, args.regular_weight)
                epoch_count = 0
                print('Start lesson:', data_len)


def evaluate_model(args, task_name, logger):
    global input_lang
    global output_lang

    input_lang, output_lang, _, pairs_test = prepare_dataset('nl', 'action', task_name)

    test_data = pairs_test
    test_data.sort(key=lambda p: len(p[0].split()))

    args.vocab_size = input_lang.n_words
    args.label_size = output_lang.n_words

    model = HRLModel(x_ratio_rate=args.simplicity_ratio,
                     encode_mode=args.encode_mode,
                     decay_r=args.decay_r,
                     vocab_size=args.vocab_size,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     label_dim=args.label_size,
                     composer_leaf=args.composer_leaf,
                     composer_trans_hidden=args.composer_trans_hidden,
                     input_lang=input_lang,
                     output_lang=output_lang).cuda(args.gpu_id)

    checkpoint_file = args.checkpoint
    print("loading", checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    print("loading finished...")
    print("Start testing ..")
    test_acc = evaluate(test_data, model, args.gpu_id)
    print("Test Acc: {} %".format(test_acc * 100))


def prepare_arguments(checkpoint_folder, parser):
    composer_lr = 1.0
    solver_lr = 0.1
    accumulate_batch_size = 4
    regular_weight = 1e-4
    regular_decay_rate = 0.5
    hidden_size = 128
    encode_mode = 'seq'

    args = {"word-dim": hidden_size,
            "hidden-dim": hidden_size,
            "composer_leaf": "no_transformation",
            "composer-trans-hidden": hidden_size,
            "regular-weight": regular_weight,  # 0.0001
            "clip-grad-norm": 0.5,
            "env-optimizer": "adadelta",  # adadelta
            "pol-optimizer": "adadelta",  # adadelta
            "env-lr": composer_lr,  # 1.
            "pol-lr": solver_lr,  # 0.1
            "l2-weight": 0.0001,
            # TODO: currently the batch size must be set as 1
            #  since our implementation requires it as to be.
            #  if you want to accumulate gradients, please use accumulate_batch_size
            "batch-size": 1,
            "accumulate-batch-size": accumulate_batch_size,
            "max-epoch": 300,
            "gpu-id": 0,
            "model-dir": "checkpoint/models/" + checkpoint_folder,
            "logs-path": "checkpoint/logs/" + checkpoint_folder,
            "encode-mode": encode_mode,
            "regular-decay-rate": regular_decay_rate}

    parser.add_argument("--word-dim", required=False, default=args["word-dim"], type=int)
    parser.add_argument("--hidden-dim", required=False, default=args["hidden-dim"], type=int)
    parser.add_argument("--composer_leaf", required=False, default=args["composer_leaf"],
                        choices=["no_transformation", "lstm_transformation",
                                 "bi_lstm_transformation", "conv_transformation"])
    parser.add_argument("--composer-trans-hidden", required=False, default=args["composer-trans-hidden"], type=int)

    parser.add_argument("--clip-grad-norm", default=args["clip-grad-norm"], type=float,
                        help="If the value is less or equal to zero clipping is not performed.")

    parser.add_argument("--env-optimizer", required=False, default=args["env-optimizer"],
                        choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--pol-optimizer", required=False, default=args["pol-optimizer"],
                        choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--env-lr", required=False, default=args["env-lr"], type=float)
    parser.add_argument("--pol-lr", required=False, default=args["pol-lr"], type=float)
    parser.add_argument("--l2-weight", required=False, default=args["l2-weight"], type=float)
    parser.add_argument("--batch-size", required=False, default=args["batch-size"], type=int)
    parser.add_argument("--accumulate-batch-size", required=False, default=args["accumulate-batch-size"], type=int)

    parser.add_argument("--max-epoch", required=False, default=args["max-epoch"], type=int)
    parser.add_argument("--gpu-id", required=False, default=args["gpu-id"], type=int)
    parser.add_argument("--model-dir", required=False, default=args["model-dir"], type=str)
    parser.add_argument("--logs-path", required=False, default=args["logs-path"], type=str)
    parser.add_argument("--encode-mode", required=False, default=args["encode-mode"], type=str)

    parser.add_argument("--regular-weight", default=args["regular-weight"], type=float)
    parser.add_argument("--regular-decay-rate", required=False, default=args["regular-decay-rate"], type=float)
    parser.add_argument("--init-regular-weight", required=False, default=1e-1, type=float)
    # default no reward decay
    parser.add_argument("--decay-r", required=False, default=1.0, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", required=True, default='train',
                            choices=['train', 'test'], type=str,
                            help="Determine whether to train a model or test using a trained weight file")
    arg_parser.add_argument("--checkpoint", required=True, type=str,
                            help="When training, it is the folder to store model weights; "
                                 "Otherwise it is the weight path to be loaded.")
    arg_parser.add_argument("--task", required=True, type=str,
                            choices=["addjump", "around_right", "simple", "length",
                                     "extend", "mcd1", "mcd2", "mcd3"],
                            help="All tasks on SCAN, the task name is used to load train or test file")
    arg_parser.add_argument("--random-seed", required=False, default=1, type=int)
    arg_parser.add_argument("--simplicity-ratio", required=False, default=0.0, type=float)

    parsed_args = arg_parser.parse_args()
    if parsed_args.mode == 'train':
        args = prepare_arguments(parsed_args.checkpoint, arg_parser)
        logger, visualizer = make_path_preparations(args, parsed_args.mode)
        train_model(args, parsed_args.task, logger)
    else:
        args = prepare_arguments(parsed_args.checkpoint, arg_parser)
        logger, visualizer = make_path_preparations(args, parsed_args.mode)
        evaluate_model(args, parsed_args.task, logger)
