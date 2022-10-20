# LogiGAN
This repository serves primarily as codebase and data, model for training, evaluation and inference of the logical pre-training method LogiGAN.
[LogiGAN](https://arxiv.org/abs/2205.08794) (NeurIPS 2022) is the adversarial logical pre-training method with Transformer-based encoder-decoder backbone.
# Preprocessing
## Logic MLM Corpus Construction
```angular2
cd corpus_construction/mlm_corpus
bash construct_premise.sh
bash construct_conclusion.sh
```
## Elastic Search for External Negatives
```
cd corpus_construction/elastic_search
bash run_gen.sh
bash run_ver.sh
```
# Adversarial Pretraining
Noting that the generator and verifier should be warmed up with constructed corpus to achieve better performance.
Afterwards,
```
cd pre-training
#launcher the program, the setting of each step is adjusted in:
python launcher_es.py
(The parameters are adjusted in parameters16g_es_corpusb.py)
```
# Citation
If you find this resource useful, please cite the paper introducing LogiGAN:

```
@article{pi2022logigan,
  title={LogiGAN: Learning Logical Reasoning via Adversarial Pre-training},
  author={Pi, Xinyu and Zhong, Wanjun and Gao, Yan and Duan, Nan and Lou, Jian-Guang},
  journal={arXiv preprint arXiv:2205.08794},
  year={2022}
}
```
