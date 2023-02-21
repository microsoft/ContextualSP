import pdb
import subprocess
import argparse
import os


def run_command(bash_command):
    process = subprocess.Popen(bash_command.split())
    output, error = process.communicate()
    print(error)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="", help="model_name_or_path")
    parser.add_argument("--output_dir", type=str, default="./checkpoint/", help="output dir")
    parser.add_argument("--train_file", type=str, default='pretrain', help="train file")
    parser.add_argument("--validation_file", type=str, default='test', help="validation file")
    parser.add_argument("--max_steps", type=int, default=100000, help="max_steps")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--ebatch_size", type=int, default=16, help="eval batch_size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight_decay")
    parser.add_argument("--gas", type=int, default=1, help="gradient_accumulation_steps")
    parser.add_argument("--save_steps", type=int, default=100000, help="save_steps")
    parser.add_argument("--device_num", type=int, default=1, help="device_num")
    parser.add_argument("--method", type=str, default='MainExp', help="method")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--init_weights", type=bool, default=False, help="init_weights")
    parser.add_argument("--subtask", type=str, default='Com', help="subtask")
    parser.add_argument("--set", type=str, default='set1', help="subtask")
    args = parser.parse_args()

    print("START training")
    run_command("printenv")

    output_dir = './checkpoint/' + args.subtask + '/' + args.method + '_' + args.train_file + '_' + args.set + '_seed' + str(args.seed)
    # ./checkpoint/Com/MainExp_pretrain_set1_seed1

    train_file = '../../data/' + args.subtask + '/' + args.set + '/' + args.train_file + '.json'
    # .../data/Com/set1/pretrain.json

    validation_file = '../../data/' + args.subtask + '/' + args.set + '/' + args.validation_file + '.json'
    # .../data/Com/set1/test.json

    cmd = f"""
        python -m torch.distributed.launch --nproc_per_node {args.device_num} --master_port=12343 t5_train_model.py \
        --model_name_or_path {args.model_name_or_path} \
        --output_dir {output_dir} \
        --do_train \
        --do_eval \
        --train_file {train_file} \
        --validation_file {validation_file} \
        --per_device_train_batch_size {args.batch_size} \
        --per_device_eval_batch_size {args.ebatch_size} \
        --overwrite_output_dir \
        --gradient_accumulation_steps {args.gas} \
        --max_steps {args.max_steps} \
        --logging_steps 10 \
        --learning_rate {args.learning_rate} \
        --save_steps {args.save_steps} \
        --eval_steps {args.save_steps} \
        --evaluation_strategy steps \
        --freeze_model_parameter False \
        --weight_decay {args.weight_decay} \
        --label_smoothing_factor 0.1 \
        --lr_scheduler_type constant \
        --fp16 False \
        --predict_with_generate \
        --num_beams 5 \
        --seed {args.seed} \
        --adafactor False \
        --max_source_length 1024 \
        --max_target_length 1024 \
        --gradient_checkpointing False \
        --init_weights {args.init_weights}
        """


    print("RUN {}".format(cmd))
    run_command(cmd)
