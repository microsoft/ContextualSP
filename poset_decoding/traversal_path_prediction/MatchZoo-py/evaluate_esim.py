import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import os
import json
print('matchzoo version', mz.__version__)

split = "mcd1"
data_root = "./data/"
model_path = f"./model/traversal_path_esim-{split}"

task = mz.tasks.Classification(num_classes=2)
task.metrics = ['acc']
print("`classification_task` initialized with metrics", task.metrics)
best_model = sorted(os.listdir(model_path), key=lambda fn: os.path.getmtime(model_path+'/' + fn))[-1]

test_raw = mz.datasets.cfq.load_data(stage='test', task=task, data_root= data_root, suffix="mask_predict_classification.csv")

print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')
# print(model_path, )
preprocessor = mz.load_preprocessor(model_path)
# preprocessor.fit(train_raw)
# train_processed = preprocessor.transform(train_raw)
test_processed = preprocessor.transform(test_raw)

# print(test_processed.frame())


testset = mz.dataloader.Dataset(
    data_pack=test_processed,
    mode='point',
    batch_size=1024,
    shuffle = False
)

padding_callback = mz.models.ESIM.get_default_padding_callback()

testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='test',
    callback=padding_callback
)
model = mz.models.ESIM()

model.params['task'] = task
model.params['embedding_input_dim'] = preprocessor.context['embedding_input_dim']
model.guess_and_fill_missing_params()
model.build()
model.load_state_dict(torch.load(f"{model_path}/{best_model}"))

optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=testloader,
    validloader=testloader,
    validate_interval=None,
    epochs=50,
    save_all = False,
    save_dir=model_path,
    device=[0,1,2,3,4,5,6,7]
)
# print(trainer.evaluate(testloader))
print(len(testloader.label))
# print(len(pred))

y_pred = trainer.predict(testloader)
open(f"./output/esim-mask-{split}-predict.prob", "w").write(json.dumps(y_pred.tolist()))
y_pred = np.argmax(y_pred, axis=1)

open(f"./output/esim-mask-{split}-predict", "w").write(json.dumps(y_pred.tolist()))


assert len(y_pred) == len(testloader.label)
print(np.sum(y_pred == testloader.label) / float(len(y_pred)))
