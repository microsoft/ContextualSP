# Compositionality Generalization <img src="https://pytorch.org/assets/images/logo-dark.svg" height = "25" align=center />

This repository is the official implementation of our paper [Compositional Generalization by Learning Analytical Expressions](https://arxiv.org/pdf/2006.10627.pdf).

If you find our code useful for you, please consider citing our paper

```bib
@inproceedings{qian2020compositional,
  title={Compositional Generalization by Learning Analytical Expressions},
  author={Liu, Qian and An, Shengnan and Lou, Jian-Guang and Chen, Bei and Lin, Zeqi and Gao, Yan and Zhou, Bin and Zheng, Nanning and Zhang, Dongmei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## Content

- [Install Requirements](#requirements)
- [Train Model](#training)
- [Evaluate Model](#evaluation)
- [Pre-trained Models](#pre-trained-models)
- [Expected Results](#results)
- [Frequent Asked Questions](#faq)


## Requirements

Our code is officially supported by Python 3.7. The main dependencies are `pytorch` and `tensorboardX`.
You could install all requirements by the following command:

```console
❱❱❱ pip install -r requirements.txt
```

## Training

To train our model on different tasks on SCAN and SCAN-ext datasets, you could use this command:

```console
❱❱❱ python main.py --mode train --checkpoint <model_dir> --task <task_name>
```

📋 Note that `<model_dir>` specifies the store folder of model checkpoints, and `<task_name>` is the task name.
Available task names are `[simple, addjump, around_right, length, mcd1, mcd2, mcd3, extend]`.

For example, you could train a model on `addjump` task by the following command:

```console
❱❱❱ python main.py --mode train --checkpoint addjump_model --task addjump
```

📋 Since reinforcement learning is known to be hard to train, there is a chance of the code to not converge in the training. You could choose another random seed and try again. 

📋 Meanwhile, please note that the model training is sensitive to the value of the hyper-parameter coefficient of the **simplicity-based reward** (i.e. `--simplicity-ratio` in args). When it is higher (i.e. 0.5 or 1.0), the model is harder to converge, which indicates that the training accuracy may not arrive at 100%. We're still investigating in the reason behind it. If you cannot obtain good results after trying several random seed, you could try to reproduce other results (not suitable for `around_right` and `mcd3`, as stated in the paper) using a `0` simplicity-ratio (default setting now). We will update the code when we find a better training strategy.

Therefore, please use the following command for `around_right` and `mcd3` task:

```console
❱❱❱ python main.py --mode train --checkpoint addjump_model --task around_right --simplicity-ratio 0.5
```

The corresponding log and model weights will be stored in the path `checkpoint/logs/addjump_model.log` and `checkpoint/models/addjump_model/*.mdl` respectively

## Evaluation

To evaluate our model on different tasks, run:

```console
❱❱❱ python main.py --mode test --checkpoint <model_weight_file> --task <task_name>
```

📋 Note that `<model_weight_file>` specifies a concrete model file with the suffix `.mdl`, and `<task_name>` is the task name.

For example, you could evaluate a trained model weight `weight.mdl` on `addjump` task by the following command:

```console
❱❱❱ python main.py --mode test --checkpoint weight.mdl --task addjump
```

## Pre-trained Models

You can find pretrained model weights for the above tasks under the `pretrained_weights` folder.


## Results

Our model is excepted to achieve 100% accuracies on all tasks if the training succeeds.
