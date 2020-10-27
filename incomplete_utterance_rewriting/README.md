# Incomplete Utterance Rewriting <img src="https://pytorch.org/assets/images/logo-dark.svg" height = "25" align=center />

[中文版](README_zh.md)

The official pytorch implementation of our paper [Incomplete Utterance Rewriting as Semantic Segmentation](https://arxiv.org/pdf/2009.13166.pdf).

If you find our code useful for you, please consider citing our paper:

```bib
@inproceedings{qian2020incomplete,
  title={Incomplete Utterance Rewriting as Semantic Segmentation},
  author={Liu, Qian and Chen, Bei and Lou, Jian-Guang and Zhou, Bin and Zhang, Dongmei},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020}
}
```

## Content

- [Install Dependencies](#requirement)
- [Download and Preprocess Dataset](#data)
- [Train Model](#train)
- [Evaluate Model](#evaluate)
- [Pre-trained Models](#pre-trained-models)

## Requirement

### Python Environment

First of all, you should setup a python environment. This code base has been tested under python 3.x, and we officially support python 3.7.

After installing python 3.7, we strongly recommend you to use `virtualenv` (a tool to create isolated Python environments) to manage the python environment. You could use following commands to create a environment.

```bash
python -m pip install virtualenv
virtualenv venv
```

### Activate Virtual Environment
Then you should activate the environment to install the dependencies. You could achieve it via using the command as below. (Please change $ENV_FOLDER to your own virtualenv folder path, e.g. venv)

```bash
$ENV_FOLDER\Scripts\activate.bat (Windows)
source $ENV_FOLDER/bin/activate (Linux)
```

### Install Libraries

The most important requirements of our code base are as following:
- pytorch >= 1.2.0 (not tested on other versions, but 1.0.0 may work though)
- allennlp == 0.9.0

Other dependencies can be installed by

```console
pip install -r requirement.txt
```

## Data

### Prepare Dataset

Although we cannot provide dataset resources (copyright issue) in our repo, we provide `download.sh` for automatically downloading and preprocessing datasets used in our paper.

> Here the preprocessing does not include exporting the distant supervision, a.k.a. the word-level edit matrix, used in our paper. Anyone interested in the distant supervision can focus on the dataset reader file `src/data_reader.py (line 178-200)`. 

### Prepare Glove

If you want to train models on English datasets (i.e. `Task` and `CANARD`), please download [Glove 6B](http://nlp.stanford.edu/data/glove.6B.zip). Unzip and move the `glove.6B.100d.txt` file into the folder `glove`.

## Train

You could train models on different datasets using `*.sh` files under the `src` folder.

For example, you could train `RUN + BERT` on `Multi` by running the following command under the `src` folder as:

```console
./train_multi_bert.sh
```

### Configs Table

| Config | Model in Paper | 
| :--- | :---: |
| canard.jsonnet | RUN on CANARD (Elgohary et al. 2019) |
| multi.jsonnet | RUN on Multi (Pan et al. 2019) |
| multi_bert.jsonnet | RUN + BERT on Multi (Pan et al. 2019) |
| rewrite.jsonnet | RUN on Rewrite (Su et al. 2019) |
| rewrite_bert.jsonnet | RUN + BERT on Rewrite (Su et al. 2019) |
| task.jsonnet | RUN on Task (Quan et al. 2019) |


### Tips for training

1. If you does not rely on `BLEU` metrics to pick up your best weight file on the dev set, you could disable it to achieve a faster evaluation speed.
2. By default we do not calculate any metric on the train set to save training time, but you could enable it by setting `enable_training_log` as `True` in `*.jsonnet` (Refer readers to see `task.jsonnet`).
3. All configs are tested successfully under `Tesla M40 (24GB)`, and if there is any error such as `CUDA Out Of Memory`, you could solve it by reducing the hyper-parameter `batch_size` in `*.jsonnet`. In our experience, it will not hurt the performance by a large margin.

## Evaluate

Once a model is well trained, `allennlp` will save a compressed model zip file which is usually named after `model.tar.gz` under the checkpoint folder. Our evaluation is based on it.

We provide a evaluate file under `src` folder, and you could evaluate a model file by running the following command:

```concolse
python evaluate.py --model_file model.tar.gz --test_file ../dataset/Multi/test.txt
```

The above script will generate a file `model.tar.gz.json` which records the detailed performance. For example, the performance of `RUN + BERT` on `Rewrite` is:
```json
{
    "ROUGE": 0.9394040084189113,
    "_ROUGE1": 0.961865057419486,
    "_ROUGE2": 0.9113051224617216,
    "EM": 0.688,
    "_P1": 0.9451903332806824,
    "_R1": 0.8668694770389685,
    "F1": 0.9043373129817137,
    "_P2": 0.8648273949812838,
    "_R2": 0.7989241803278688,
    "F2": 0.8305705345849144,
    "_P3": 0.8075098814229249,
    "_R3": 0.7449860216360763,
    "F3": 0.774988935954985,
    "_BLEU1": 0.9405510823944796,
    "_BLEU2": 0.9172718486250105,
    "_BLEU3": 0.8932687251641028,
    "BLEU4": 0.8691863201601382,
    "loss": 0.2084200546145439
}
```
Next, we will provide all pre-trained models to reproduce results reported in our paper. We recommend you to download them and put them into the folder `pretrained_weights` and run commands like below:

```concolse
python evaluate.py --model_file ../pretrianed_weights/rewrite.tar.gz --test_file ../dataset/Multi/test.txt
```

## Pre-trained Models


| Dataset | BERT | Config | EM | Rewriting F1 | BLEU4 | Pretrained_Weights |
| :---: | :---: |:--- | :---: | :---: | :---: | :---: |
| Rewrite | No | rewrite.jsonnet | 53.6 | 81.3 | 79.6 | [rewrite.tar.gz](https://github.com/microsoft/ContextualSP/releases/download/rewrite/rewrite.tar.gz)|
| Rewrite | Yes | rewrite_bert.jsonnet | 68.8 | 90.4 | 86.9 | [rewrite_bert.tar.gz](https://github.com/microsoft/ContextualSP/releases/download/rewrite.bert/rewrite_bert.tar.gz)|
| CANARD | No | canard.jsonnet | 18.3 | 44.2 | 49.8 | [canard.tar.gz](https://github.com/microsoft/ContextualSP/releases/download/canard/canard.tar.gz) |
| Multi | No | multi.jsonnet | 43.3 | 60.7 | 81.1 | [multi.tar.gz](https://github.com/microsoft/ContextualSP/releases/download/multi/multi.tar.gz) |
| Multi | Yes | multi_bert.jsonnet | 49.2 | 70.3 | 82.5 | [multi_bert.tar.gz](https://github.com/microsoft/ContextualSP/releases/download/multi.bert/multi_bert.tar.gz) |
