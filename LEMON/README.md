# LEMON

This repository contains the code and pre-trained models for our paper [LEMON: Language-Based Environment Manipulation via Execution-guided Pre-training](https://arxiv.org/pdf/2201.08081.pdf)

Data
-------
The data is in the [release](https://github.com/microsoft/ContextualSP/releases/tag/lemon_data). Please unzip it and put it in the lemon_data folder.

Pre-training
-------
Run the following command to preprocess the data:
```bash
bash preprocess_pretrain.bat
```

Then run the following command to pre-train the model:
```bash
bash pretrain.sh
```

Fine-tuning
-------

Run the following command to preprocess the data:
```bash
bash preprocess_finetune.bat
```

Then run the following command to fine-tune the model:
```bash
bash finetune.sh
```
