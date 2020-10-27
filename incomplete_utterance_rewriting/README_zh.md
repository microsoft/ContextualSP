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

### 配置表

| 配置文件名 | 论文中对应设置 | 
| :--- | :---: |
| canard.jsonnet | RUN on CANARD (Elgohary et al. 2019) |
| multi.jsonnet | RUN on Multi (Pan et al. 2019) |
| multi_bert.jsonnet | RUN + BERT on Multi (Pan et al. 2019) |
| rewrite.jsonnet | RUN on Rewrite (Su et al. 2019) |
| rewrite_bert.jsonnet | RUN + BERT on Rewrite (Su et al. 2019) |
| task.jsonnet | RUN on Task (Quan et al. 2019) |


### 训练小提示

1. 如果读者并不依赖`BLEU`度量来得到开发集上表现最佳的权重文件，你可以禁用它来实现更快的评测速度。
2. 默认情况下，我们不会在训练集上计算任何指标以节省训练时间，但你可以通过在`*.jsonnet`中设置`enable_training_log`为`True`来启用它（请读者参考`task.jsonnet`）。
3. 所有模型的训练和评测均在`Tesla M40 (24GB)`下测试通过，如果在读者本机中出现诸如`CUDA Out Of Memory`之类的错误，读者可通过降低 `*.jsonnet` 中的超参数 `batch_size` 来解决。 根据我们的经验，这将不会对性能造成很大的影响。

## 评测模型

当模型的训练正常结束时，`allennlp`将保存一个压缩的模型文件，该文件通常以checkpoint文件夹下的`model.tar.gz`命名，我们后续的评估就基于该压缩文件。

我们在src文件夹下提供了一个用于模型评测的脚本`evaluate.py`，读者可以通过运行以下命令来评估模型文件：

```concolse
python evaluate.py --model_file model.tar.gz --test_file ../dataset/Multi/test.txt
```

上面的脚本将生成一个文件`model.tar.gz.json`，其中记录了模型详细的指标。 例如，`RUN + BERT`在`Rewrite`的性能为：
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

接下来，我们将提供所有预训练好的模型文件，以重现论文中报告的结果。 我们建议读者下载它们并将其放入`pretrained_weights`文件夹中，然后运行以下命令：
```concolse
python evaluate.py --model_file ../pretrianed_weights/rewrite.tar.gz --test_file ../dataset/Multi/test.txt
```


## 预训练模型权重

请注意，提供的基于BERT的预训练模型**明显优于**论文报告的结果。

| Dataset | BERT | Config | EM | Rewriting F1 | BLEU4 | Pretrained_Weights |
| :---: | :---: |:--- | :---: | :---: | :---: | :---: |
| Rewrite | No | rewrite.jsonnet | 53.6 | 81.3 | 79.6 | [rewrite.tar.gz](https://github.com/microsoft/ContextualSP/releases/download/rewrite/rewrite.tar.gz)|
| Rewrite | Yes | rewrite_bert.jsonnet | 68.8 | 90.4 | 86.9 | [rewrite_bert.tar.gz](https://github.com/microsoft/ContextualSP/releases/download/rewrite.bert/rewrite_bert.tar.gz)|