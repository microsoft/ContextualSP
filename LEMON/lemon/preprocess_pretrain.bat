DATASET_PATH=path_to_dataset
MODEL_PATH=path_to_bart_large
python -m bpe_encoder \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/train.src \
            --outputs $DATASET_PATH/train.bpe.src \
            --workers 20 \
            --keep-empty
python -m bpe_encoder \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/train.tgt \
            --outputs $DATASET_PATH/train.bpe.tgt \
            --workers 20 \
            --keep-empty

python -m bpe_encoder \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/dev.src \
            --outputs $DATASET_PATH/dev.bpe.src \
            --workers 20 \
            --keep-empty
python -m bpe_encoder \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/dev.tgt \
            --outputs $DATASET_PATH/dev.bpe.tgt \
            --workers 20 \
            --keep-empty

fairseq-preprocess --source-lang "src" --target-lang "tgt" \
    --trainpref $DATASET_PATH/train.bpe \
    --validpref $DATASET_PATH/dev.bpe \
    --destdir $DATASET_PATH/bin_large \
    --workers 20 \
    --srcdict $MODEL_PATH/dict.txt \
    --tgtdict $MODEL_PATH/dict.txt