export CUDA_VISIBLE_DEVICES=2

python t5_run_train.py \
--model_name_or_path t5-base \
--subtask Com \
--method ControlExp \
--train_file finetune \
--max_steps 50000 \
--save_steps 50000 \
--batch_size 8 \
--ebatch_size 16 \
--gas 1 \
--seed 1 \
--set set1