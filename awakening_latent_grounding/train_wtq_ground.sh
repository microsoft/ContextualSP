python train.py \
    -model WTQAlignmentModel \
    -bert bert-base-uncased \
    -lr 5e-5 \
    -train_bs 24 \
    -alw linear_20-50 \
    -num_epochs 50  \
    --data_dir data/wtq_grounding \
    --out_dir checkpoints/wtq_grounding_model
