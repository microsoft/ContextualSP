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

logger = logging.getLogger(__name__)


def set_train_parser(parser_group):
    train_parser = parser_group.add_parser("train")
    train_parser.add_argument("--dataset-dir", type=str, required=True, default="",
                              help="dataset directory where train.src is located in")
    train_parser.add_argument("--exp-dir", type=str, default="checkpoints",
                              help="experiment directory which stores the checkpoint weights")
    train_parser.add_argument("--model-path", type=str, default="tapex.base/model.pt",
                              help="the directory of pre-trained model path")
    train_parser.add_argument("--model-arch", type=str, default="bart_base", choices=["bart_large", "bart_base"],
                              help="tapex large should correspond to bart_large, and tapex base should be bart_base")
    train_parser.add_argument("--max-tokens", type=int, default=1536,
                              help="if you train a large model on 16GB memory, max-tokens should be empirically "
                                   "set as 1536, and can be near-linearly increased according to your GPU memory.")
    train_parser.add_argument("--gradient-accumulation", type=int, default=8,
                              help="the accumulation steps to arrive a equal batch size, the default value can be used"
                                   "to reproduce our results. And you can also reduce it to a proper value for you.")
    train_parser.add_argument("--total-num-update", type=int, default=10000,
                              help="the total optimization training steps")
    train_parser.add_argument("--learning-rate", type=float, default=3e-5,
                              help="the peak learning rate for model training")
    train_parser.add_argument("--warmup-steps", type=int, default=1500,
                              help="warmup steps")
    train_parser.add_argument("--wandb-project", type=str, default='universal_pretrain_bart',
                              help="wandb-project")


def set_eval_parser(parser_group):
    eval_parser = parser_group.add_parser("eval")
    eval_parser.add_argument("--dataset-dir", type=str, required=True, default="",
                             help="dataset directory where train.src is located in")
    eval_parser.add_argument("--model-path", type=str, default="tapex.base.wikisql/model.pt",
                             help="the directory of fine-tuned model path such as tapex.base.wikisql/model.pt")
    eval_parser.add_argument("--sub-dir", type=str, default="valid", choices=["train", "valid", "test"],
                             help="the directory of pre-trained model path, and the default should be in"
                                  "{bart.base, bart.large, tapex.base, tapex.large}.")
    eval_parser.add_argument("--max-tokens", type=int, default=1536 * 4,
                             help="the max tokens can be larger than training when in inference.")
    eval_parser.add_argument("--predict-dir", type=str, default="predict",
                             help="the predict folder of generated result.")


def set_predict_parser(parser_group):
    predict_parser = parser_group.add_parser("predict")
    predict_parser.add_argument("--resource-dir", type=str, required=True, default="./tapex.base",
                                help="the resource dir which contains the model weights, vocab.bpe, "
                                     "dict.src.txt, dict.tgt.txt and encoder.json.")
    predict_parser.add_argument("--checkpoint-name", type=str, default="model.pt",
                                help="the model weight's name in the resource directory")


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
        --label-smoothing 0.1  \
        --max-source-positions 1024 \
        --max-tokens {args.max_tokens}  \
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
        --save-interval-updates	2000 \
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
        --max-tokens {args.max_tokens} \
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


# def predict_demo(args):
#     demo_interface = TAPEXModelInterface(resource_dir=args.resource_dir,
#                                          checkpoint_name=args.checkpoint_name)
#     question = "Greece held its last Summer Olympics in which year?"
#     table_context = {
#         "header": ["Year", "City", "Country", "Nations"],
#         "rows": [
#             [1896, "Athens", "Greece", 14],
#             [1900, "Paris", "France", 24],
#             [1904, "St. Louis", "USA", 12],
#             [2004, "Athens", "Greece", 201],
#             [2008, "Beijing", "China", 204],
#             [2012, "London", "UK", 204]
#         ]
#     }
#     answer = demo_interface.predict(question=question,
#                                     table_context=table_context)
#     logger.info("Receive question as : {}".format(question))
#     logger.info("The answer should be : {}".format(answer))


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    set_train_parser(subparsers)
    set_eval_parser(subparsers)
    set_predict_parser(subparsers)

    args = parser.parse_args()
    if args.subcommand == "train":
        train_fairseq_model(args)
    elif args.subcommand == "eval":
        evaluate_fairseq_model(args)
    elif args.subcommand == "predict":
        predict_demo(args)
