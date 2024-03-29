# README

The official code of paper [Awakening Latent Grounding from Pretrained Language Models for Semantic Parsing](https://aclanthology.org/2021.findings-acl.100.pdf).

# Install Dependencies

Please first install [PyTorch](https://pytorch.org/), and then install all the dependencies by running:

```bash
pip install -r requirements.txt
```

Please remember to unzip the `json.zip` in the `data/wtq_grounding` folder. And the file structure should be like:

```bash
data/wtq_grounding
├── json
│   ├── 202.json
│   ├── 203.json
│   ├── ...
├── dev.json
├── test.json
└── ...
```

# Train Grounding Model

## Train Grounding Model on Spider

Please run the script `train_spider_ground.sh` to train the grounding model on Spider dataset.

## Train Grounding Model on WTQ

Please run the script `train_wtq_ground.sh` to train the grounding model on WTQ dataset.

# Evaluate Grounding Model

## Evaluate Grounding Model on Spider

Please run the script `eval_spider_ground.sh` to evaluate the grounding model on Spider dataset. Note that you should replace the model checkpoint `checkpoints/spider_grounding_model/model.pt` with yours.

You should get the following results after following the training script:

```bash
avg loss = 0.2189                                                                                                 
table accuracy = 0.8453                                                                                           
column accuracy = 0.7602                                                                                          
value accuracy = 0.9449                                                                                           
overall accuracy = 0.7050                                                                                         
table  P = 0.847, R = 0.857, F1 = 0.852                                                                           
column  P = 0.842, R = 0.838, F1 = 0.840                                                                          
value  P = 0.948, R = 0.932, F1 = 0.940                                                                           
average F1 = 0.8773                                                                                               
```

## Evaluate Grounding Model on WTQ

Please run the script `eval_wtq_ground.sh` to evaluate the grounding model on WTQ dataset.  Note that you should replace the model checkpoint `checkpoints/wtq_grounding_model/model.pt` with yours.
