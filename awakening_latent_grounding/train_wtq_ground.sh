python train.py -model WTQAlignmentModel -bert bert-base-uncased \
    -lr 3e-5 -train_bs 16 -alw linear_5-10 -num_epochs 50  \
    --data_dir data/wtq_grounding \
    --out_dir checkpoints/model_wtq