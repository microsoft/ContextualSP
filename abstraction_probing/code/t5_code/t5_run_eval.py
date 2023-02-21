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
    parser.add_argument("--validation_file", type=str, default='test', help="validation file")
    parser.add_argument("--ebatch_size", type=int, default=16, help="eval batch_size")
    parser.add_argument("--device_num", type=int, default=1, help="device_num")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="checkpoint_step")
    # parser.add_argument("--log_metrics", type=str, default='False', help="log_metrics")
    # parser.add_argument("--log_label", type=str, default='False', help="log_label")
    # parser.add_argument("--log_metrics_only", type=str, default='False', help="log_metrics_only")
    parser.add_argument("--num_beams", type=int, default=5, help="num_beams")
    parser.add_argument("--subtask", type=str, default='Com', help="subtask")
    parser.add_argument("--set", type=str, default='set1', help="subtask")
    # parser.add_argument("--with_constraint", type=str, default='True', help="with_constraint")
    args = parser.parse_args()

    print("START training")
    run_command("printenv")

    validation_file = '../../data/' + args.subtask + '/' + args.set + '/' + args.validation_file + '.json'
    # .../data/Com/set1/test.json

    cmd = f"""
        python -m torch.distributed.launch --nproc_per_node {args.device_num} --master_port=12345 t5_eval_model.py \
        --model_name_or_path {args.model_name_or_path} \
        --output_dir {args.output_dir} \
        --do_eval \
        --validation_file {validation_file} \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size {args.ebatch_size} \
        --overwrite_output_dir \
        --gradient_accumulation_steps 1 \
        --max_steps 100000 \
        --logging_steps 10 \
        --learning_rate 1e-5 \
        --save_steps 100000 \
        --eval_steps 100000 \
        --evaluation_strategy steps \
        --freeze_model_parameter \
        --weight_decay 1e-2 \
        --label_smoothing_factor 0.1 \
        --lr_scheduler_type constant \
        --fp16 False \
        --predict_with_generate \
        --dev_split -1 \
        --num_beams {args.num_beams} \
        --seed 1 \
        --adafactor False \
        --max_source_length 1024 \
        --max_target_length 1024 \
        --log_metrics True \
        --log_label True \
        --eval_type test \
        --log_metrics_only False \
        --with_constraint True
        """

    print("RUN {}".format(cmd))
    run_command(cmd)
