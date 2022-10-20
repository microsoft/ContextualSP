MODEL_PATH=${1}
SAVE_PATH=${2}
GPU_NUM=16
python -m torch.distributed.launch --nproc_per_node ${GPU_NUM} verifier_multi_es.py --model_path ${MODEL_PATH} --output_dir ${SAVE_PATH}
#python verifier_multi_es.py --model_path ${MODEL_PATH} --output_dir ${SAVE_PATH}
