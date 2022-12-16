# README

The official code of paper [Awakening Latent Grounding from Pretrained Language Models for Semantic Parsing](https://aclanthology.org/2021.findings-acl.100.pdf).

# Install Dependencies

Please first install [PyTorch](https://pytorch.org/), and then install all the dependencies by running:

```bash
pip install -r requirements.txt
```

# Train Grounding Model

## Train Grounding Model on Spider

Please run the script `train_spider_ground.sh` to train the grounding model on Spider dataset.

## Train Grounding Model on WTQ

Please run the script `train_wtq_ground.sh` to train the grounding model on WTQ dataset.

# Evaluate Grounding Model

## Evaluate Grounding Model on Spider

Please run the script `eval_spider_ground.sh` to evaluate the grounding model on Spider dataset. Note that you should replace the model checkpoint `checkpoints/spider_grounding_model/model.pt` with yours.

## Evaluate Grounding Model on WTQ

Please run the script `eval_wtq_ground.sh` to evaluate the grounding model on WTQ dataset.  Note that you should replace the model checkpoint `checkpoints/wtq_grounding_model/model.pt` with yours.
