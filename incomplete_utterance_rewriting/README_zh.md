# 不完整话语重写 <img src="https://pytorch.org/assets/images/logo-dark.svg" height = "25" align=center />

[English Version](README.md)

本仓库是论文[Incomplete Utterance Rewriting as Semantic Segmentation](https://arxiv.org/pdf/2009.13166.pdf)的官方实现。在这篇论文中，我们将*不完整话语重写*任务视为一个面向对话编辑的任务，并据此提出一个全新的、使用语义分割思路来解决该任务的模型。

如果本仓库或论文对您的研究有所帮助，请考虑使用以下bibtex引用我们的论文:

```bib
@inproceedings{qian2020incomplete,
  title={Incomplete Utterance Rewriting as Semantic Segmentation},
  author={Liu, Qian and Chen, Bei and Lou, Jian-Guang and Zhou, Bin and Zhang, Dongmei},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020}
}
```

## 目录

- [依赖安装](#依赖安装)
- [数据集下载与预处理](#数据集下载与预处理)
- [训练模型](#训练模型)
- [评测模型](#评测模型)
- [预训练模型权重](#预训练模型权重)

## 依赖安装

### Python 环境


首先，你应该设置一个python环境。本仓库理论上可以在python 3.x下直接运行，我们实验中所使用到的是 python 3.7。在安装了python 3.7之后，我们强烈建议你使用 `virtualenv`（一个创建独立Python环境的工具）来管理python环境。你可以使用以下命令来创建环境：

```bash
python -m pip install virtualenv
virtualenv venv
```

### 激活虚拟环境

在安装完虚拟环境后，你需要激活环境才能安装本仓库依赖的库。你可以通过使用下面的命令来安装 (注意需要把 $ENV_FOLDER 改为你自己的 virtualenv 文件夹路径，例如 venv)：

```bash
$ENV_FOLDER\Scripts\activate.bat (Windows)
source $ENV_FOLDER/bin/activate (Linux)
```

### 安装依赖

本仓库最主要的两个依赖库是`pytorch`和`allennlp`，其版本需求如下:
- pytorch >= 1.2.0 (没有在其他版本上测试过，但1.0.0可能可以用)
- allennlp == 0.9.0

其他所有依赖都可以通过以下命令安装:

```console
pip install -r requirement.txt
```

## 数据集下载与预处理

### 准备数据集

虽然我们不能在本仓库中直接提供数据集（因为版权问题），但我们提供了`download.sh`用于自动下载和预处理论文中所用到的数据集。

> 值得注意的是，对数据集的预处理过程不包括导出论文中使用的远端监督(Distant Supervision)数据，也就是词级别的编辑矩阵。对该处理流程感兴趣的读者可以关注文件`src/data_reader.py（第178-200行)`。

### 准备Glove文件

如果你想在英文数据集（即`Task`和`CANARD`）上训练模型，需要下载[Glove 6B 词向量](http://nlp.stanford.edu/data/glove.6B.zip)。解压该文件，并将`glove.6B.100d.txt`文件移动到`glove`文件夹中。

## 训练模型

你可以使用`src`文件夹下的`*.sh`文件在不同的数据集上训练模型。例如，你可以在`src`文件夹下运行以下命令，以在`Multi`数据集上训练`RUN + BERT`模型。

```console
./train_multi_bert.sh
```

> 训练时的一些提示。
> - 如果读者并不依赖`BLEU`度量来得到开发集上表现最佳的权重文件，你可以禁用它来实现更快的评测速度。
> - 默认情况下，我们不会在训练集上计算任何指标以节省训练时间，但你可以通过在`*.jsonnet`中设置`enable_training_log`为`True`来启用它（请读者参考`task.jsonnet`）。

## 评测模型

[TODO]

## 预训练模型权重

[TODO]