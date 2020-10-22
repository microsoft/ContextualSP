# Compositionality Generalization <img src="https://pytorch.org/assets/images/logo-dark.svg" height = "25" align=center />

This repository is the official implementation of our paper [Compositional Generalization by Learning Analytical Expressions](https://arxiv.org/pdf/2006.10627.pdf).

> Code for reproducing MinSCAN results is being transferred from the experimental codebase into here, please stay tuned. 

## Citation

If you find our code useful for you, please consider citing our paper

```latex
@inproceedings{qian2020compositional,
  title={Compositional Generalization by Learning Analytical Expressions},
  author={Liu, Qian and An, Shengnan and Lou, Jian-Guang and Chen, Bei and Lin, Zeqi and Gao, Yan and Zhou, Bin and Zheng, Nanning and Zhang, Dongmei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## Requirements

Our code is officially supported by Python 3.7. The main dependencies are `pytorch` and `tensorboardX`.
You could install all requirements by the following command:

```setup
pip install -r requirements.txt
```

## Training

To train our model on different tasks on SCAN and SCAN-ext datasets, you could use this command:

```train
python main.py --mode train --checkpoint <model_dir> --task <task_name>
```

📋 Note that `<model_dir>` specifies the store folder of model checkpoints, and `<task_name>` is the task name.
Available task names are `[simple, addjump, around_right, length, mcd1, mcd2, mcd3, extend]`.

For example, you could train a model on `addjump` task by the following command:

```train
python main.py --mode train --checkpoint addjump_model --task addjump
```

The corresponding log and model weights will be stored in the path `checkpoint/logs/addjump_model.log` and `checkpoint/models/addjump_model/*.mdl` respectively

## Evaluation

To evaluate our model on different tasks, run:

```eval
python main.py --mode test --checkpoint <model_weight_file> --task <task_name>
```

📋 Note that `<model_weight_file>` specifies a concrete model file with the suffix `.mdl`, and `<task_name>` is the task name.

For example, you could evaluate a trained model weight `weight.mdl` on `addjump` task by the following command:

```eval
python main.py --mode test --checkpoint weight.mdl --task addjump
```

## Pre-trained Models

You can find pretrained model weights for the above tasks under the `pretrained_weights` folder.


## Results

Our model is excepted to achieve 100% accuracies on all tasks if the training succeeds.
