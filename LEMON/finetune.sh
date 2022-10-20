python lemon/run_model_finetune.py \
    --dataset-dir lemon_data/dataset/DATASET_PREFIX/bin_large \
    --exp-dir OUTPUT_PATH \
    --model-path PRE_TRAINED_MODEL_PATH \
    --model-arch bart_large \
    --total-num-update 10000 \
    --batch-size 64 \
    --gradient-accumulation 1 \
    --warmup-steps 1500 \
    --learning-rate 3e-5