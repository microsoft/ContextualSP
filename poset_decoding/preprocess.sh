

# Local path to the dataset (after it has been downloaded).
dataset_local_path="./data/dataset.json"


if [[ ! -f "${dataset_local_path}"  ]]; then
  echo "ERROR: Dataset not found."
  echo "Please download the dataset first from ${dataset_url}!"
  echo "See further instructions in the README."
  exit 1
fi

# preprocess data from raw CFQ dataset.json
python -m preprocess_cfq --dataset_path="${dataset_local_path}" \
  --split_path="./data/splits/mcd1.json" --save_path="./data/mcd1/"

python -m preprocess_cfq --dataset_path="${dataset_local_path}" \
  --split_path="./data/splits/mcd2.json" --save_path="./data/mcd2/"

python -m preprocess_cfq --dataset_path="${dataset_local_path}" \
  --split_path="./data/splits/mcd3.json" --save_path="./data/mcd3/"

# preprocess data for sketch
python preprocess_hierarchical_training.py

# preprocess data for primi