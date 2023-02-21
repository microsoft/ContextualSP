export CUDA_VISIBLE_DEVICES=3

python t5_run_train.py \
--model_name_or_path t5-base \
--subtask Com \
--method ContrastExp \
--train_file pretrain_contrast \
--max_steps 100000 \
--save_steps 100000 \
--batch_size 8 \
--ebatch_size 16 \
--gas 1 \
--seed 1 \
--set set1