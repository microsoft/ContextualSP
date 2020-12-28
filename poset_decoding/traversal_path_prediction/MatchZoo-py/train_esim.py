import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import os
print('matchzoo version', mz.__version__)


model_name = "esim-mcd1"
model_path = f"../../model/traversal_path_{model_name}/"
data_path = f"../../data/"
if not os.path.exists(model_path):
	os.mkdir(model_path)

task = mz.tasks.Classification(num_classes=2)
task.metrics = ['acc']
print("`classification_task` initialized with metrics", task.metrics)

# task = mz.tasks.Classification()
train_raw = mz.datasets.cfq.load_data(stage='train', task=task, data_root=data_path, suffix="mask_classification.csv")
dev_raw = mz.datasets.cfq.load_data(stage='dev', task=task, data_root=data_path, suffix = "mask_classification.csv")


print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

preprocessor = mz.models.ESIM.get_default_preprocessor()
train_processed = preprocessor.fit_transform(train_raw)
if os.path.exists(f"{model_path}/preprocessor.dill"):
	os.remove(f"{model_path}/preprocessor.dill")
preprocessor.save(model_path)
dev_processed = preprocessor.transform(dev_raw)

print(train_processed.frame())
print(dev_processed.frame())

trainset = mz.dataloader.Dataset(
    data_pack=train_processed,
    mode='point',
    batch_size = 256,
    shuffle = True
)
devset = mz.dataloader.Dataset(
    data_pack=dev_processed,
    mode='point',
    batch_size=256,
    shuffle = False
)

padding_callback = mz.models.ESIM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
devloader = mz.dataloader.DataLoader(
    dataset=devset,
    stage='dev',
    callback=padding_callback
)

model = mz.models.ESIM()

model.params['task'] = task
model.params['embedding_input_dim'] = preprocessor.context['embedding_input_dim']
model.guess_and_fill_missing_params()
model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=devloader,
    validate_interval=None,
    epochs=50,
    save_all = False,
    save_dir=model_path,
    device=[0,1,2, 3,4,5,7]
)

trainer.run()

