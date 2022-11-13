# Environment Setup

```
conda create -n grounding python=3.7  -y
conda activate grounding
conda install pytorch==1.7.1 cudatoolkit=10.1 -c pytorch
pip install -r  requirements.txt
```

# Training

```shell
python train.py -exp_id _experiment_ -datasets wikisql_label -use_wb -threads 16 -plm bert-base-uncased -model UniG -bs 25 -ls 0.05 -out_dir grounding
```

# Inference

```shell
python infer.py -ckpt  checkpoint/xxx.pt  -gpu
```