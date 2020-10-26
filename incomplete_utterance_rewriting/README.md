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

> Some tips for training:
> - If you does not rely on `BLEU` metrics to pick up your best weight file on the dev set, you could disable it to achieve a faster evaluation speed.
> - By default we do not calculate any metric on the train set to save training time, but you could enable it by setting `enable_training_log` as `True` in `*.jsonnet` (Refer readers to see `task.jsonnet`).

## Evaluate

[TODO]

## Pre-trained Models

[TODO]