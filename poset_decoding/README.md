## Poset Decoding <img src="https://pytorch.org/assets/images/logo-dark.svg" height = "25" align=center />

The official pytorch implementation of our paper [Hierarchical Poset Decoding for Compositional Generalization in Language](https://arxiv.org/pdf/2002.00652.pdf). 

If you find our code useful, please consider citing our paper:

```
@inproceedings{Yinuo2020Hirearchical,
  title={Hierarchical Poset Decoding for Compositional Generalization in Language},
  author={Yinuo Guo and Zeqi Lin and Jian-Guang Lou and Dongmei Zhang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```


## Dependency

pip install -r requirements.txt

## Data preprocess

get CFQ data : Download dataset from [link](https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz)

```bash
bash preprocess.sh
```

## Training

###	Sketch Prediction

```bash
cd sketch_prediction/
bash ./train.sh
```

### Traversal Path Prediction 

> The module is based on the open-source project Matchzoo-py <https://github.com/NTMC-Community/MatchZoo-py>

```bash
cd  ./traversal_path_prediction/MatchZoo-py/
python ./traversal_path_prediction/MatchZoo-py/train_esim.py
```

## Evaluation

```bash
bash evaluate.sh
```

## MCD2 and MCD3

In the aforementioned Training and Evaluation sections, we train and evaluate HPD on the MCD1 split.

To train and evaluate on MCD2/MCD3 split, please replace ``mcd1'' to ``mcd2'' or ``mcd3'' in the following files:

- sketch_prediction/train.sh
- sketch_prediction/evaluate.sh
- traversal_path_prediction/MatchZoo-py/train_esim.py
- traversal_path_prediction/MatchZoo-py/evaluate_esim.py
- traversal_path_prediction/MatchZoo-py/datasets/cfq/load_data.py
- evaluate.sh


##  Acknowledgement

We will thank the following repos which are very helpful to us.
- [Matchzoo-py](https://github.com/NTMC-Community/MatchZoo-py)

## Contact

Any question please contact `zeqi DOT lin AT microsoft DOT com`