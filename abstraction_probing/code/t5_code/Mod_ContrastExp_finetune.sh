export CUDA_VISIBLE_DEVICES=6

python t5_run_train.py \
--model_name_or_path ./checkpoint/Mod/ContrastExp_pretrain_contrast_set1_seed1/checkpoint-100000 \
--subtask Mod \
--method ContrastExp \
--train_file finetune \
--max_steps 50000 \
--save_steps 50000 \
--batch_size 8 \
--ebatch_size 16 \
--gas 1 \
--seed 1 \
--set set1