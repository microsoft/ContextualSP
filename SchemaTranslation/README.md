# Schema Translation 

This repository is the official implementation of our paper [Translating Headers of Tabular Data: A Pilot Study of Schema Translation](https://aclanthology.org/2021.emnlp-main.5.pdf).

If you find our code useful for you, please consider citing our paper.

```bib
@inproceedings{zhu-etal-2021-translating,
    title = "Translating Headers of Tabular Data: A Pilot Study of Schema Translation",
    author = "Zhu, Kunrui  and
      Gao, Yan  and
      Guo, Jiaqi  and
      Lou, Jian-Guang",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.5",
    pages = "56--66",
}
```
## Content

- [Dataset](#dataset)
- [Install Requirements](#requirements)
- [Train Model](#training)
- [Evaluate Model](#evaluation)
- [Pre-trained Models](#pre-trained-models)

## Dataset
In this work, we a parallel dataset for schema translation, which consists of 3,158 tables with 11,979 headers written in 6 different languages, including English, Chinese, French, German, Spanish, and Japanese.
Our tables are collected from three resources. Firstly, we use all 2,108 tables from the [WikiTableQuestion](https://github.com/ppasupat/WikiTableQuestions) dataset. Secondly, we manually collect 176 English tables from the search engine covering multiple domains like retail, education, and government. At last, we select all the tables from the [Spider](https://github.com/taoyds/spider) dataset. Finally, we obtained 3,158 tables with 11,979 headers in total.

The dataset/ directory contains the orignal English schema and their translation results in Chinese, French, German, Spanish, and Japanese. 
The following jsonl format is an example from the validation set
```
{"table_name": "203-csv\\43.tsv", "en": "Polling Firm", "zh": "民意调查公司", "de": "Abfrage Firma", "fr": "Société d'enquête", "ja": "世論調査会社", "es": "Empresa encuestadora"}
{"table_name": "203-csv\\43.tsv", "en": "Link", "zh": "附件", "de": "Anhang", "fr": "Lien", "ja": "リンク", "es": "Enlace"}

```
## Requirements
The required python packages for model training can be installed by running the following commands.
```
pip install -r requirement.txt
```
The python packages and third-party tools can be downloaded and installed by running the following commands.
```
rm -rf eval
unzip eval.zip
cd eval && bash install_dependencies.sh
```

## Training
The baseline NMT models(H2H and H2H+CXT) is placed at run_Concat.py, and the implementation of CAST is placed at run_CAST.py.<br>
An example of the full pipeline of training and evaluation can be found under the notebooks/ directory.<br>
The following is the details of the training parameters. 
Taking the training process of CAST initialized with [M2M-100](https://huggingface.co/facebook/m2m100_418M) for translating English headers to German as an example.
The training process can be launched by running the following commands.
```
pretrained_model="facebook/m2m100_418M"  # The reference of the pretrained language models. In the paper facebook/m2m100_418M and facebook/mbart-large-50-many-to-many-mmt are used
tgt="de"  # The target-language symbol for the selected language model (Chinese - zh/ zh_CN), French - fr/ fr_XX etc.).
tgt_lang="de" # The fixed target-language symbol zh, fr, de, es, and ja for Chinese, French, German, Spanish, and Japanese repectively.
data_format="column+schema+val-to-column"  # One of the data format between column-to-column, column+schema+val-to-column for header to header training and using header and the corresponding context as input
max_source_length=128  # The max input length
warmup=0.2  # The warmup rate
batch_size=4  # The batch size
lr=3e-5  # The leaning rate
num_epoch=4  # The number of training epochs
model_class="M2M100RATTypeNStructureSepForST"  # The name of the model 
output_suffix=model_class+"-CAST"  # The self-defined prefix for the outputs
num_rat_layers=2  
encoder_factor=4

# train
!python run_CAST.py \
      --model_name_or_path $pretrained_model \
      --do_train \
      --do_eval \
      --do_predict \
      --source_lang=en \
      --target_lang=$tgt_lang \
      --encoder_factor $encoder_factor \
      --train_file=data/$data_format/train.jsonl \
      --validation_file=data/$data_format/valid.jsonl \
      --test_file=data/$data_format/test.jsonl \
      --output_dir=outputs/$data_format/$tgt/$output_suffix/ \
      --max_source_length=$max_source_length \
      --per_device_train_batch_size=$batch_size \
      --per_device_eval_batch_size=$batch_size \
      --learning_rate=$lr --warmup_ratio=$warmup --num_train_epochs=$num_epoch  \
      --save_strategy=epoch --evaluation_strategy=epoch \
      --overwrite_output_dir \
      --model_class=$model_class --num_rat_layers $num_rat_layers \
      --load_best_model_at_end=True \
      --predict_with_generate
```

## Evaluation
To evaluate the results of the above training process, run the following commands
```
root_dir="../"
# Tokenize the reference file following the setting of "Beyond English-Centric Multilingual Machine Translation"
!cd eval && cat "$root_dir"/data/"$data_format"/test_ref".$tgt" | sh tok.sh "$tgt" > \
  "$root_dir"/outputs/"$data_format"/"$tgt"/"$output_suffix"/ref.tok
# Tokenize the predicted file
!cd eval && cat "$root_dir"/outputs/"$data_format"/"$tgt"/"$output_suffix"/test_generations.txt | sh tok.sh "$tgt" > \
  "$root_dir"/outputs/"$data_format"/"$tgt"/"$output_suffix"/hyp.tok
# Compute the BLUE4 Score
!cd eval && python sacrebleu.py -tok 'none' --score-only \
  "$root_dir"/outputs/"$data_format"/"$tgt"/"$output_suffix"/ref.tok \
  < "$root_dir"/outputs/"$data_format"/"$tgt"/"$output_suffix"/hyp.tok \
  > "$root_dir"/outputs/"$data_format"/"$tgt"/"$output_suffix"/bleu4
!cd eval && cat "$root_dir"/outputs/"$data_format"/"$tgt"/"$output_suffix"/bleu4
```
## Pre-trained Models
The pretrained model of CAST initialized with [M2M-100](https://huggingface.co/facebook/m2m100_418M) for en-de translation can be found outputs/
- H2H: outputs/column-to-column/de/M2M100ForST-H2H/
- H2H+CXT: outputs/column+schema+val-to-column/de/M2M100ForST-H2H+CXT 
- CAST: outputs/column+schema+val-to-column/deM2M100RATTypeNStructureSepForST-CAST