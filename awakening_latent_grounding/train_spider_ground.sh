python train.py -model SpiderAlignmentModel -bert bert-large-uncased-whole-word-masking \
    -lr 5e-5 -train_bs 6 \
    -acc_steps 4 -alw linear_20-50 -num_epochs 50  \
    --data_dir data/spider_grounding \
    --out_dir checkpoints/model_spider \
    --warmup_steps 2000