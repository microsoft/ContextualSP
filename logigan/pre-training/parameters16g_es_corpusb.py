import os
############ General Parameters ##############
gan_alpha=0.8
mode='debug'
project_dir = os.getenv('HOME')#"/home/v-xinyupi/"
code_dir=f"{project_dir}/LogicPretrain/code/GAN-new"
model_name = 'LogiGAN'
corpus_dir=f"{project_dir}/LogicPretrain/Logic_gan_new/data/corpus_gan_new/beta"
model_output_dir=f"{project_dir}/LogicPretrain/models/GAN/{model_name}"
initial_gen_path=#warmpup-generator-dir
initial_ver_path=#warmup-verifier-dir
#'albert-large-v2'#f"{project_dir}/v-wanzho/LogicPretrain/models/bart_checkpoints_warmup/checkpoint-20000/"
max_iter = 15 if mode!='debug' else 1
run_dir=f'run-{model_name}'
os.makedirs(os.path.join(corpus_dir,run_dir),exist_ok=True)
############ Generator Parameters ##############
# PATH
gen_train_src_file="gen_train_src_es.jsonl"
gen_val_src_file=f"{run_dir}/gen_train_iter.jsonl"
gen_train_iter_file=f"{run_dir}/gen_train_iter.jsonl"
if mode=='debug':
    gen_train_src_file="gen_train_src_es_toy.jsonl" # DEBGU ONLY
gen_train_src_toy_file="gen_train_src_es_toy.jsonl" # DEBGU ONLY
#gen_val_src_file="gen_valid_toy.jsonl" # DEBGU ONLY
#gen_train_iter_file=f"{run_dir}/gen_train_iter_toy.jsonl" # DEBGU ONLY
unlabeled_gen_train_iter_file=f"{run_dir}/gen_train_iter_unlabeled.jsonl"  # Generator adhoc self sampling
unlabeled_ver_train_iter_file=f"{run_dir}/ver_train_iter_unlabled.jsonl" # Generator adhoc inference  for gan ver


gen_train_src_path=os.path.join(corpus_dir, gen_train_src_file)
gen_val_src_path=os.path.join(corpus_dir, gen_val_src_file)
gen_train_iter_path=os.path.join(corpus_dir, gen_train_iter_file) # Xinyu: Infered by verifier
unlabeled_gen_train_iter_path=os.path.join(corpus_dir, unlabeled_gen_train_iter_file)
unlabeled_ver_train_iter_path=os.path.join(corpus_dir, unlabeled_ver_train_iter_file)
gen_output_dir=os.path.join(model_output_dir, "gen_checkpoints")
gen_train_samples_per_iter=100000 if mode!='debug' else 100 ### Xinyu: Self-Sampling Size. Default 1e5 i.e., 10% of 100w.
# Trainer
gen_per_device_train_batch_size=# To be adjusted by GPU memory size.
gen_per_device_examples_num=# To be adjusted by GPU memory size. the number of pos+neg per batch e.g., if 1 pos, 5 neg, then it should be 6
gen_per_device_eval_batch_size=# To be adjusted by GPU memory size.
gen_gradient_accumulation_steps=8

gen_learning_rate=5e-5
# Beam search
gen_num_beams=5
num_return_seq=5
gen_max_length=256
gen_min_length=5
gen_length_penalty=4.0
gen_early_stopping=True
gen_no_repeat_ngram_size=3



############ Verifier Parameters ##############
# PATH
ver_train_src_file="ver_train_es.jsonl"
ver_train_iter_file=f"{run_dir}/ver_train_iter.jsonl"
#ver_train_src_file="ver_train_src_toy.jsonl"   ## DEBUG ONLY
ver_train_src_path=os.path.join(corpus_dir, ver_train_src_file)
# ver_train_iter_path=os.path.join(corpus_dir, ver_train_iter_file)
ver_train_iter_path=os.path.join(corpus_dir,unlabeled_ver_train_iter_file)
ver_script_path=os.path.join(code_dir, "verifier.py")
ver_output_dir=os.path.join(model_output_dir, "ver_checkpoints")
ver_train_samples_per_iter=80000 if mode!='debug' else 80# Xinyu: Default 2.7e5 i.e., ~10% of 270w

# Trainer
ver_per_device_train_batch_size=# To be adjusted by GPU memory size.
ver_per_device_eval_batch_size=# To be adjusted by GPU memory size.
ver_gradient_accumulation_steps=1
ver_learning_rate=1e-5


############ NLI Labeler Parameters ##############
nli_script_path=os.path.join(code_dir, "labeler.py")
nli_output_dir=os.path.join(model_output_dir, "labeler_checkpoints")  # this one is just a placeholder and should be empty. 
nli_per_device_eval_batch_size=24

