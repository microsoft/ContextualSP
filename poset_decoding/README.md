## Poset Decoding <img src="https://pytorch.org/assets/images/logo-dark.svg" height = "25" align=center />

The official pytorch implementation of our paper [Hierarchical Poset Decoding for Compositional Generalization in Language](https://arxiv.org/pdf/2002.00652.pdf). 

If you find our code useful, please consider citing our paper:

```
@inproceedings{Yinuo2020Hirearchical,
  title={Hierarchical Poset Decoding for Compositional Generalization in Language},
  author={Yinuo Guo and Zeqi Lin and Jian-Guang Lou and Dongmei Zhang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}```


#### Dependency

pip install -r requirements.txt

#### Data preprocess

get CFQ data : Download dataset from [link](https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz)

```bash
bash preprocess.sh
```

#### Training

##### 	Sketch Prediction

​		cd sketch_prediction/

​		bash ./train.sh

##### 	Traversal Path Prediction (The module is  based on the open-source project Matchzoo-py <https://github.com/NTMC-Community/MatchZoo-py>)

​		cd  ./traversal_path_prediction/

​		python ./traversal_path_prediction/train_esim.py

#### Evaluation

​	bash evaluate.sh


####  Acknowledgement

We will thank the following repos which are very helpful to us.
- [Fairseq](https://github.com/pytorch/fairseq)

zeqi.lin@microsoft.com