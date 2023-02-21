export CUDA_VISIBLE_DEVICES=5

python t5_run_eval.py \
--model_name_or_path ./checkpoint/Mod/ControlExp_finetune_set1_seed1/checkpoint-50000 \
--subtask Mod \
--validation_file test \
--ebatch_size 16 \
--set set1