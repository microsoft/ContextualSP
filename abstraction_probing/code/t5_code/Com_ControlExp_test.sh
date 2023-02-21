export CUDA_VISIBLE_DEVICES=2

python t5_run_eval.py \
--model_name_or_path ./checkpoint/Com/ControlExp_finetune_set1_seed1/checkpoint-50000 \
--subtask Com \
--validation_file test \
--ebatch_size 16 \
--set set1