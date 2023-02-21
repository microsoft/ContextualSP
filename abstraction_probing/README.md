# Does Deep Learning Learn to Abstract?

This is the official repo for the paper 
*'Does Deep Learning Learn to Abstract? A Systematic Probing Framework'*.
This work has been accepted at ICLR 2023.

[OpenReview](https://openreview.net/forum?id=QB1dMPEXau5)

This repo contains data and main code used in this work.
We hope this work can facilitate understanding of the abstraction capability of deep learning model.


## Data

```shell
|-- data
    |-- Com
        |-- set1
            |-- pretrain.json
            |-- finetune.json
            |-- test.json
            |-- pretrain_contrast.json
        |-- set2
        |-- set3
    |-- Mod
```

`./data` contains our two probing tasks Com and Mod.
Each probing task contain 3 different sets.
The difference among sets is that they use different terminals.
Our reported results are averaged on 3 sets.

Each set contain 4 data files.
Each line in the file is one example that has an `input` sequence and `output` sequence.
`pretrain.json` is for MainExp pretraning.
`pretrain_contrast.json` is for ContrastExp pretraining.
`finetune.json` and `test.json` is for finetuning and testing in all three Exps.

## Code

We provide the code for T5 models.
Code for GPT2 models is on the way.

### Requirements

The main dependency is `pytorch` and `transformers`.

```bash
pip install -r requirements.txt
```

### MainExp

```bash
sh Com_MainExp_pretrain.sh
```

This will start training the T5-Base model on `./data/Com/set1/pretrain.json`.
You can change the subtask, subset, and other hyper-parameters 
in `Com_MainExp_pretrain.sh` and `t5_run_train.py`.

After the training finished, the model will be saved in `/code/t5_code/checkpoint/Com/MainExp_pretrain_set1_seed1/checkpoint-100000/`.

```bash
sh Com_MainExp_finetune.sh
```

This will load the pretrained checkpoint and finetune on `./data/Com/set1/finetune.json`.

The model will be saved in `./code/t5_code/checkpoint/Com/MainExp_finetune_set1_seed1/checkpoint-100000/`.

```bash
sh Com_MainExp_test.sh
```

This will test the finetuned model on `./data/Com/set1/test.json`.

The testing results will be logged in `./code/t5_code/checkpoint/Com/MainExp_finetune_set1_seed1/checkpoint-50000_test_beam5.txt`


### ControlExp

```bash
sh Com_ControlExp_finetune.sh
sh Com_ControlExp_test.sh
```

### ContrastExp

```bash
sh Com_ContrastExp_pretrain.sh
sh Com_ContrastExp_finetune.sh
sh Com_ContrastExp_test.sh
```



## Citation

```bibtex
@inproceedings{
    an2023does,
    title={Does Deep Learning Learn to Abstract? A Systematic Probing Framework},
    author={Shengnan An and Zeqi Lin and Bei Chen and Qiang Fu and Nanning Zheng and Jian-Guang Lou},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=QB1dMPEXau5}
}
```
