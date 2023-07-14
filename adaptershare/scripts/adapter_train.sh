#!/bin/bash
usage() {
  echo "Usage: ${0} [-g|--gpu_num] [-o|--output_dir] [-m|--model_dir] [-tr|--train_datasets] [-te|--test_datasets] [-ls|--log_step] [-ss|--save_step]" 1>&2
  exit 1 
}
while [ $# -gt 0 ]
do
  key=${1}
  case ${key} in
    -g|--gpu_num)
      GPU_NUM=${2}
      shift 2
      ;;
    -o|--output_dir)
      OUT_DIR=${2}
      shift 2
      ;;
    -m|--model_dir)
      M_DIR=${2}
      shift 2
      ;;
    -tr|--train_datasets)
      TRAIN=${2}
      shift 2
      ;;
    -te|--test_datasets)
      TEST=${2}
      shift 2
      ;;
    -ls|--log_step)
      LOG_STEP=${2}
      shift 2
      ;;
    -ss|--save_step)
      SAVE_STEP=${2}
      shift 2
      ;;
    *)
      usage
      shift
      ;;
  esac
done

N_GPUS="1"

if [ ! -z "$GPU_NUM" ]; then
    N_GPUS=$GPU_NUM
fi

if [ ! -z "$TRAIN" ]; then
    TRAIN=$TRAIN
fi

if [ ! -z "$TEST" ]; then
    TEST=$TEST
fi

if [ ! -z "$LOG_STEP" ]; then
    LOG_STEP=$LOG_STEP
fi

if [ ! -z "$SAVE_STEP" ]; then
    SAVE_STEP=$SAVE_STEP
fi

NOW=$(date +"%Y%m%d%H%M")

ADAPTER_DIR="/mnt/chenzhi/checkpoints/nlu_downstream/single_task/${TRAIN}"
OUTPUT_DIR=$ADAPTER_DIR
if [ ! -z "$OUT_DIR" ]; then
    OUTPUT_DIR=$OUT_DIR
fi

MODEL_DIR="bert-large-uncased"
if [ ! -z "$M_DIR" ]; then
    MODEL_DIR=$M_DIR
fi


echo "Run Name: $RUN_NAME"
echo "Model Dir:" $MODEL_DIR
echo "Output Dir:" $OUTPUT_DIR

Run_Command_Args=" --init_checkpoint $MODEL_DIR"
Run_Command_Args="$Run_Command_Args --train_datasets ${TRAIN}"
Run_Command_Args="$Run_Command_Args --test_datasets ${TEST}"
Run_Command_Args="$Run_Command_Args --log_per_updates $LOG_STEP"
Run_Command_Args="$Run_Command_Args --save_per_updates_on true"
Run_Command_Args="$Run_Command_Args --save_per_updates $SAVE_STEP"
Run_Command_Args="$Run_Command_Args --epochs 10"
Run_Command_Args="$Run_Command_Args --batch_size 8"
Run_Command_Args="$Run_Command_Args --batch_size_eval 8"
Run_Command_Args="$Run_Command_Args --grad_accumulation_step 2"
Run_Command_Args="$Run_Command_Args --output_dir $OUTPUT_DIR"



echo $Run_Command_Args
CUDA_VISIBLE_DEVICES=0 python adapter_train.py $Run_Command_Args