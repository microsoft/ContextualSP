import subprocess
import argparse
import os


def run_command(bash_command):
    process = subprocess.Popen(bash_command.split())
    output, error = process.communicate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="dataset path")
    parser.add_argument("--exp_name", type=str, default="", help="test")
    parser.add_argument("--models_path", type=str, default="", help="models path")
    parser.add_argument("--bart_model_path", type=str, default="", help="bart init models path")
    parser.add_argument("--total_num_update", type=int, default=200000)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensorboard_path", type=str, default="", help="tensorboard path")
    args = parser.parse_args()

    print("START training")
    run_command("printenv")

    restore_file = os.path.join(args.bart_model_path, "model.pt")

    cmd = f"""
    fairseq-train {args.dataset_path} \
    --save-dir {args.models_path}/{args.exp_name} \
    --restore-file {restore_file} \
    --arch bart_large  \
    --criterion label_smoothed_cross_entropy  \
    --source-lang src  \
    --target-lang tgt  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens {args.max_tokens}  \
    --update-freq 4  \
    --max-update {args.total_num_update}  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.05  \
    --optimizer adam  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 1e-05  \
    --total-num-update {args.total_num_update}  \
    --warmup-updates 5000  \
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
    --validate-interval-updates 500 \
    --validate-interval	10 \
    --save-interval	10 \
    --patience 200 \
    --no-last-checkpoints \
    --no-save-optimizer-state \
    --report-accuracy 
    """

    print("RUN {}".format(cmd))
    run_command(cmd)
