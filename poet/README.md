# POET

This is the official repo for the paper [Reasoning Like Program Executors](https://arxiv.org/pdf/2201.11473.pdf).

## Pre-training Corpus

You can find the pre-training SQL corpus from [here](https://drive.google.com/file/d/1dg3NwPT2vWTcj2rx7S6GN8x5EywZiXQr), the pre-training Math corpus from [here](https://huggingface.co/datasets/SivilTaram/poet-math). 

The pre-training SQL corpus can be synthesized following the same procedure as done in [TAPEX](https://github.com/microsoft/Table-Pretraining#-synthesize-your-own-pre-training-data) with the `expand_numbers_in_text` function below:

```python
def expand_numbers_in_text(text, delim=" ", ignore_chars=[","], reverse_num=False):
    number_pattern = r"[-+]?[.]?[\d]+(,\d+)*[\.]?\d*(?:[eE][-+]?\d+)?%?"
    num_char_spans = [(m.start(0), m.end(0)) for m in re.finditer(number_pattern, text)]
    if len(num_char_spans) == 0: return text
    out_text = ""
    last_e = -1
    for i, (s, e) in enumerate(num_char_spans):
        out_text += text[:s] if i == 0 else text[last_e:s]
        num_str = delim.join([c for c in list(text[s:e]) if c not in ignore_chars])
        out_text += num_str if not reverse_num else num_str[::-1]
        last_e = e
    out_text += text[last_e:]  # append rest
    return out_text
```

The pre-training Math corpus can be synthesized by the script [synthesize_math_corpus.py](synthesize_math_corpus.py).
The pre-training Logic corpus can be synthesized by the script [synthesize_logic_corpus.py](synthesize_logic_corpus.py).

For all BART-based experiments, we use the [fairseq](https://github.com/facebookresearch/fairseq) implementation, which means that you can prepare the dataset as the following format:
```
|- dataset
    |- train.src
    |- train.tgt
    |- valid.src
    |- valid.tgt
```

After necessary preprocessing (you can follow the official guide in fairseq machin translation task), you can use the following script to train the model:

```shell
fairseq-train dataset/bin/ \
    --save-dir models \
    --tensorboard-logdir tensorboard_logs \
    --restore-file BART-large/model.pt \
    --arch bart_large \
    --task translation \
    --maximize-best-checkpoint-metric \
    --criterion label_smoothed_cross_entropy  \
    --source-lang src  \
    --target-lang tgt  \
    --label-smoothing 0.1  \
    --max-tokens 1536 \
    --validate-interval	50 \
    --save-interval	50 \
    --save-interval-updates	3001 \
    --validate-interval-updates 3001 \
    --keep-interval-updates 5 \
    --update-freq 16 \
    --warmup-updates 500  \
    --max-update 20000  \
    --total-num-update 20000  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01  \
    --optimizer adam  \
    --adam-betas "(0.9, 0.999)"  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 3e-5  \
    --ddp-backend no_c10d  \
    --num-workers 1  \
    --reset-meters  \
    --reset-optimizer \
    --reset-dataloader \
    --share-all-embeddings \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 10  \
    --patience 10 \
    --keep-best-checkpoints 1 \
    --report-accuracy \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --no-save-optimizer-state
```

## Pre-trained Model Weights

You can find all the available POET model weights at [Huggingface Hub](https://huggingface.co/models?search=siviltaram/poet).
For all these models, you can try to fine-tune them as the vanilla models. And these models are pre-trained on the following format of `natural context` and `sentence`:

```
[sentence] col : [natural context]
```

where `[sentence]` is usually the question in the task, and `[natural context]` is usually the passage in the task. Please refer to our paper for more details.