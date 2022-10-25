# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from argparse import ArgumentParser
from fairseq_cli.train import cli_main as fairseq_train
from fairseq_cli.generate import cli_main as fairseq_generate
import logging
import shlex
import re
import os
sys.path.append('../')
# from model_interface import TAPEXModelInterface
from model_eval import evaluate_generate_file
from collections import Counter

logger = logging.getLogger(__name__)


def set_parser(parser):
    parser.add_argument("--dataset-dir", type=str, required=True, default="",
                              help="dataset directory where train.src is located in")
    parser.add_argument("--exp-dir", type=str, default="checkpoints",
                              help="experiment directory which stores the checkpoint weights")
    parser.add_argument("--model-path", type=str, default="tapex.base/model.pt",
                              help="the directory of pre-trained model path")
    parser.add_argument("--model-arch", type=str, default="bart_base", choices=["bart_large", "bart_base"],
                              help="tapex large should correspond to bart_large, and tapex base should be bart_base")
    # train_parser.add_argument("--max-tokens", type=int, default=1536,
    #                           help="if you train a large model on 16GB memory, max-tokens should be empirically "
    #                                "set as 1536, and can be near-linearly increased according to your GPU memory.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                              help="the accumulation steps to arrive a equal batch size, the default value can be used"
                                   "to reproduce our results. And you can also reduce it to a proper value for you.")
    parser.add_argument("--total-num-update", type=int, default=20000,
                              help="the total optimization training steps")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                              help="the peak learning rate for model training")
    parser.add_argument("--warmup-steps", type=int, default=1500,
                              help="warmup steps")
    parser.add_argument("--seed", type=int, default=1,
                              help="random seed")
    parser.add_argument("--wandb-project", type=str, default='universal_pretrain_bart',
                              help="wandb-project")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                              help="label smoothing")
    parser.add_argument("--sub-dir", type=str, default="valid", choices=["train", "valid", "test"],
                             help="the directory of pre-trained model path, and the default should be in"
                                  "{bart.base, bart.large, tapex.base, tapex.large}.")
    parser.add_argument("--predict-dir", type=str, default="predict",
                             help="the predict folder of generated result.")


def train_fairseq_model(args):
    cmd = f"""
        fairseq-train {args.dataset_dir} \
        --save-dir {args.exp_dir} \
        --restore-file {args.model_path} \
        --arch {args.model_arch}  \
        --memory-efficient-fp16	\
        --task translation  \
        --criterion label_smoothed_cross_entropy  \
        --source-lang src  \
        --target-lang tgt  \
        --truncate-source  \
        --label-smoothing {args.label_smoothing}  \
        --max-source-positions 1024 \
        --batch-size {args.batch_size}  \
        --update-freq {args.gradient_accumulation} \
        --max-update {args.total_num_update}  \
        --required-batch-size-multiple 1  \
        --dropout 0.1  \
        --attention-dropout 0.1  \
        --relu-dropout 0.0  \
        --weight-decay 0.01  \
        --optimizer adam  \
        --adam-eps 1e-08  \
        --clip-norm 0.1  \
        --lr-scheduler polynomial_decay  \
        --lr {args.learning_rate}  \
        --total-num-update {args.total_num_update}  \
        --warmup-updates {args.warmup_steps}  \
        --seed {args.seed} \
        --ddp-backend no_c10d  \
        --num-workers 20  \
        --reset-meters  \
        --reset-optimizer \
        --reset-dataloader \
        --share-all-embeddings \
        --layernorm-embedding \
        --share-decoder-input-output-embed  \
        --skip-invalid-size-inputs-valid-test  \
        --log-format json  \
        --log-interval 10  \
        --save-interval-updates	500 \
        --validate-interval	50 \
        --save-interval	50 \
        --patience 200 \
        --report-accuracy \
        --wandb-project {args.wandb_project}
    """
    sys.argv = shlex.split(cmd)
    logger.info("Begin to train model for dataset {}".format(args.dataset_dir))
    logger.info("Running command {}".format(re.sub("\s+", " ", cmd.replace("\n", " "))))
    fairseq_train()


def evaluate_fairseq_model(args):
    cmd = f"""
        fairseq-generate 
        --path {args.model_path} \
        {args.dataset_dir} \
        --truncate-source \
        --gen-subset {args.sub_dir} \
        --batch-size {args.batch_size}  \
        --nbest 1 \
        --source-lang src \
        --target-lang tgt \
        --results-path {args.predict_dir} \
        --beam 5 \
        --bpe gpt2 \
        --remove-bpe \
        --num-workers 20 \
        --skip-invalid-size-inputs-valid-test
    """
    sys.argv = shlex.split(cmd)
    logger.info("Begin to evaluate model on the {} subset of dataset {}".format(args.sub_dir, args.dataset_dir))
    logger.info("Running command {}".format(re.sub("\s+", " ", cmd.replace("\n", " "))))
    fairseq_generate()
    # after generation, we should call TAPEX evaluate function to evaluate the result
    generate_file = os.path.join(args.predict_dir, "generate-{}.txt".format(args.sub_dir))
    # the delimiter is the answer delimiter used in training, which by default is a comma
    evaluate_generate_file(generate_file, target_delimiter=", ")


def eval_all_checkpoints(args):

    for args.sub_dir in ['valid', 'test']:
        all_checkpoint_name_list = [item for item in list(os.listdir(args.exp_dir)) if item.endswith('.pt')]
        print(all_checkpoint_name_list)
        print('{} checkpoints needs to evaluate'.format(len(all_checkpoint_name_list)))
        for model_name in all_checkpoint_name_list:
            args.model_path = os.path.join(args.exp_dir, model_name)
            args.predict_dir = args.model_path[:-3]
            evaluate_fairseq_model(args)

def post_eval(eval_file, data_file):

    eval_lines = open(eval_file, 'r').readlines()[1:]
    data_lines = open(data_file, 'r').readlines()

    result_1utts_list = []
    result_2utts_list = []
    result_3utts_list = []
    result_4utts_list = []
    result_5utts_list = []

    for line in eval_lines:
        # print(line)
        result, _, _, source, id = line.strip().split('\t')
        assert source.strip() == data_lines[int(id)].strip()
        if int(id) % 5 == 0:
            result_1utts_list.append(result)
        elif int(id) % 5 == 1:
            result_2utts_list.append(result)
        elif int(id) % 5 == 2:
            result_3utts_list.append(result)
        elif int(id) % 5 == 3:
            result_4utts_list.append(result)
        elif int(id) % 5 == 4:
            result_5utts_list.append(result)


    result_1utts = Counter(result_1utts_list)
    result_2utts = Counter(result_2utts_list)
    result_3utts = Counter(result_3utts_list)
    result_4utts = Counter(result_4utts_list)
    result_5utts = Counter(result_5utts_list)
    result_1utts = result_1utts['True'] / sum(result_1utts.values())
    result_3utts = result_3utts['True'] / sum(result_3utts.values())
    result_5utts = result_5utts['True'] / sum(result_5utts.values())
    return round(result_1utts,3), round(result_3utts,3), round(result_5utts,3)

def post_eval_with_generated_file(args):

    result_1utts_dict = {}
    result_3utts_dict = {}
    result_5utts_dict = {}
    all_checkpoint_name_list = [item for item in list(os.listdir(args.exp_dir)) if item.endswith('.pt')]
    print('{} checkpoints needs to evaluate'.format(len(all_checkpoint_name_list)))
    for model_name in all_checkpoint_name_list:
        model_path = os.path.join(args.exp_dir, model_name)
        predict_dir = model_path[:-3]
        eval_file = os.path.join(predict_dir, 'generate-valid.txt.eval')
        data_file = os.path.join(args.dataset_dir, '../dev.src')
        result_1utts, result_3utts, result_5utts = post_eval(eval_file, data_file)
        print("path: {}, stage: {}, 1utts: {}, 3utts: {}, 5utts: {}".format(model_path, 'valid', result_1utts, result_3utts, result_5utts))
        result_1utts_dict[model_path] = result_1utts
        result_3utts_dict[model_path] = result_3utts
        result_5utts_dict[model_path] = result_5utts
        
        eval_file_test = os.path.join(predict_dir, 'generate-test.txt.eval')
        data_file_test = os.path.join(args.dataset_dir, '../test.src')
        result_1utts_test, result_3utts_test, result_5utts_test = post_eval(eval_file_test, data_file_test)
        print("path: {}, stage: {}, 1utts: {}, 3utts: {}, 5utts: {}".format(model_path, 'test', result_1utts_test, result_3utts_test, result_5utts_test))
        print('~~~')

    best_key = max(result_5utts_dict, key=result_5utts_dict.get)
    print(best_key)
    print(result_1utts_dict[best_key])
    print(result_3utts_dict[best_key])
    print(result_5utts_dict[best_key])
    print('**************************************************************')


if __name__ == '__main__':
    parser = ArgumentParser()
    set_parser(parser)

    args = parser.parse_args()

    train_fairseq_model(args)
    eval_all_checkpoints(args)
    post_eval_with_generated_file(args)
