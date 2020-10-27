# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import random
import re

import jieba
import spacy
from tqdm import tqdm

random.seed(42)
nlp_en = spacy.load('en_core_web_sm')


def is_all_chinese(word):
    # identify whether all chinese characters
    for _char in word:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def cut_mixed_sentence(text):
    # for chinese, return character; for english, return word;
    jieba_words = list(jieba.cut(text))
    ret_chars = []
    for word in jieba_words:
        if is_all_chinese(word):
            ret_chars.extend(list(word))
        else:
            ret_chars.append(word)
    return ' '.join(ret_chars)


def cut_english_sentence(text):
    text = re.sub('\t\t', ' ', text)
    doc = nlp_en(text)
    ret_words = []
    for word in doc:
        if word.text.strip():
            ret_words.append(word.text.lower())
    return ' '.join(ret_words)


def unified_dataset_format(dataset_id):
    if dataset_id == 'Rewrite':
        origin_file = "corpus.txt"
        with open(origin_file, "r", encoding="utf8") as f:
            total_lines = [line.strip() for line in f.readlines()]
            total_len = len(total_lines)

            border = int(0.9 * total_len)
            train_data = total_lines[:border]
            dev_data = total_lines[border:]

            for train_ind in range(len(train_data)):
                sentences = train_data[train_ind].split('\t\t')
                new_sen = []
                for sentence in sentences:
                    new_sen.append(cut_mixed_sentence(sentence))
                train_data[train_ind] = '\t\t'.join(new_sen)

            for dev_ind in range(len(dev_data)):
                sentences = dev_data[dev_ind].split('\t\t')
                new_sen = []
                for sentence in sentences:
                    new_sen.append(cut_mixed_sentence(sentence))
                dev_data[dev_ind] = '\t\t'.join(new_sen)

            with open("train.txt", "w", encoding="utf8") as train_f:
                train_f.write('\n'.join(train_data))

            with open("dev.txt", "w", encoding="utf8") as dev_f:
                dev_f.write('\n'.join(dev_data))
    elif dataset_id == 'Multi':
        src_files = ["train.sr",
                     "valid.sr",
                     "test.sr"]
        tgt_files = ["train.tr",
                     "valid.tr",
                     "test.tr"]
        for src_file, tgt_file in zip(src_files, tgt_files):
            src_f = open(src_file, "r", encoding="utf8")
            tgt_f = open(tgt_file, "r", encoding="utf8")
            src_lines = src_f.readlines()
            tgt_lines = tgt_f.readlines()

            # WARNING: there is an annotation bug in test.sr 3224
            if 'test' in src_file:
                actual_line = src_lines[3222].split("\t")[0]
                src_lines[3222] = actual_line + ' 已 经 玩 过 了 |\n'
                del src_lines[3223]

            dataset = []
            for src_line, tgt_line in zip(src_lines, tgt_lines):
                src_line = src_line.strip('\n')
                tgt_line = tgt_line.strip()

                valid_sen = src_line[:src_line.rfind('|')].strip()
                border_pos = valid_sen.rfind(' || ')
                context_str, cur_str = valid_sen[:border_pos], valid_sen[border_pos + 4:]
                context_str = context_str.replace(' <split> ', '\t\t')
                context_str += '\t\t' + cur_str + '\t\t' + tgt_line
                dataset.append(context_str)

            modes = ['train', 'valid', 'test']
            write_path = None
            for sample_mode in modes:
                if sample_mode in src_file:
                    write_path = sample_mode + ".txt"
                    break
            with open(write_path, "w", encoding="utf8") as write_f:
                write_f.write('\n'.join(dataset))
            write_f.close()
    elif dataset_id == 'CANARD':
        src_files = ["train.json",
                     "dev.json",
                     "test.json"]
        for src_file in src_files:
            content = json.load(open(src_file, "r", encoding="utf8"))
            dataset = []
            for example in tqdm(content):
                sent_history = '\t\t'.join([cut_english_sentence(sen)
                                            for sen in example['History']])
                incomplete_sent = cut_english_sentence(example['Question'])
                rewrite_sent = cut_english_sentence(example['Rewrite'])
                context_str = sent_history + '\t\t' + incomplete_sent + '\t\t' + rewrite_sent
                dataset.append(context_str)
            modes = ['train', 'dev', 'test']
            write_path = None
            for sample_mode in modes:
                if sample_mode in src_file:
                    write_path = sample_mode + ".txt"
                    break
            with open(write_path, "w", encoding="utf8") as write_f:
                write_f.write('\n'.join(dataset))
            write_f.close()
    elif dataset_id == 'Task':
        src_file = "CamRest676_annotated.json"
        with open(src_file, "r", encoding="utf8") as f:
            content = json.load(f)
            dataset = []
            example_border = 0
            for dialogue in tqdm(content):
                sent_history = []
                for example in dialogue['dial']:
                    context_str = '\t\t'.join(sent_history[-2:])
                    if context_str == '':
                        # Just a placeholder
                        context_str = 'hello'
                    complete_str = cut_english_sentence(example['usr']['transcript_complete'])
                    cur_is_incomplete = False
                    case_number = 0
                    if example['usr']['transcript_with_ellipsis'] != "":
                        cur_is_incomplete = True
                        dataset.append('\t\t'.join([context_str,
                                                    cut_english_sentence(example['usr']['transcript_with_ellipsis']),
                                                    complete_str]))
                        case_number += 1
                    # TODO: follow the original setting which only considers part of corpus
                    elif example['usr']['transcript_with_coreference'] != "":
                        cur_is_incomplete = True
                        dataset.append('\t\t'.join([context_str,
                                                    cut_english_sentence(example['usr']['transcript_with_coreference']),
                                                    complete_str]))
                        case_number += 1
                    if not cur_is_incomplete:
                        dataset.append('\t\t'.join([context_str,
                                                    complete_str,
                                                    complete_str]))
                        case_number += 1
                    sent_history.append(cut_english_sentence(complete_str))
                    sent_history.append(cut_english_sentence(example['sys']['sent']))
                    if dialogue['dialogue_id'] < 540:
                        example_border += case_number
            # shuffle dataset
            train_data = dataset[:example_border]
            dev_data = dataset[example_border:]
            with open("train.txt", "w", encoding="utf8") as train_f:
                train_f.write('\n'.join(train_data))

            with open("dev.txt", "w", encoding="utf8") as dev_f:
                dev_f.write('\n'.join(dev_data))
    else:
        raise Exception("We do not support it currently!")


if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--dataset", required=True,
    #                         choices=['Task', 'Rewrite', 'Multi', "CANARD"], type=str,
    #                         help="Please specify a dataset you want to process")
    # parsed_args = arg_parser.parse_args()
    # unified_dataset_format(parsed_args.dataset)
    unified_dataset_format("Multi")
