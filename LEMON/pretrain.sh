python lemon/run_model_pretrain.py train \
    --dataset-dir lemon_data/pretraining_corpus/DATASET_PREFIX/bin_large \
    --exp-dir OUTPUT_PATH \
    --model-path BART_MODEL_PATH \
    --model-arch bart_large \
    --total-num-update 10000 \
    --max-tokens 1800 \
    --gradient-accumulation 8 \
    --warmup-steps 1500 
    