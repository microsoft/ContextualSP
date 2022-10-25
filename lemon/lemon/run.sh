# pretrain
python run_model_pretrain.py train --dataset-dir PRETRAINING_CORPUS_DIR --exp-dir OUTPUT_DIR --model-path BART_LARGE_PATH --model-arch bart_large --total-num-update 10000 --max-tokens 1800 --warmup-steps 1500
# finetune
python run_model_finetune.py --dataset-dir DATASET_DIR --exp-dir OUTPUT_DIR --model-path PRETRAINED_MODEL_PATH --model-arch bart_large --total-num-update 10000 --batch-size 64 --gradient-accumulation 1 --warmup-steps 1500 --learning-rate 3e-5