import subprocess
import argparse
import os, sys, time
import torch
from parameters16g_es_corpusb import *

## Xinyu: Copied from run_hf.sh, remember to change CUDA_VISIBLE_DEVICES & nproc_per_node
# --model_name_or_path $1 \
# --output_dir $2 \
# --data_dir $3 \
# --train_file $4 \
# --validation_file $5 \
# --pad_to_max_length\
# --per_device_train_batch_size $6 \
# --gradient_accumulation_steps $7 \
# --learning_rate $8 \
# --num_train_epochs $9 \
# --seed ${10} \
# --remove_unused_columns False \
# --num_beams 4 \
# --save_strategy epoch\
# --evaluation_strategy no \
# --logging_steps 200 \
# --max_train_samples ${11} \
# --max_predict_samples ${12} \
# --predict_with_generate \
# --do_predict ${13} \
# --test_file ${14} \
# --do_eval False \
# --do_train ${15} \
# --prediction_mode ${16} \
# --overwrite_cache\
# --overwrite_output_dir



sh_parameters_gen = {
    1: None,
    2: None,
    3: corpus_dir,
    4: gen_train_iter_file,
    5: gen_val_src_file,
    6: gen_per_device_train_batch_size,
    7: gen_gradient_accumulation_steps,
    8: gen_learning_rate,
    9: 1, # num of epoch by default to 1
    10: None,
    11: gen_train_samples_per_iter, 
    12: None,
    13: None,
    14: None,
    15: None,
    16: None,
    17:None,
    18:None,
    19:gen_per_device_eval_batch_size
}

def print_cmd_launch(cmd, iter, info):
    print(f"\n\n\n=================  iter {iter}   ============================")
    print(f"{info}\n")
    print(f"Launching cmd:\n{cmd}\n")
    print(f"========================================================\n\n\n")

def run_command(bash_command):
    try:
        process = subprocess.Popen(bash_command.split())
        process.wait()
        output, error = process.communicate()
        print(error)
        print(output)
    except Exception as e:
        exit()


def build_step0_cmd(gen_model_path,alpha):
    sh_parameters_gen[1] = gen_model_path
    sh_parameters_gen[2] = gen_output_dir  # inference only
    sh_parameters_gen[4] = gen_train_src_toy_file
    sh_parameters_gen[10] = int(time.time())
    sh_parameters_gen[12] = gen_train_samples_per_iter
    sh_parameters_gen[13] = True
    sh_parameters_gen[14] =  gen_train_src_file
    sh_parameters_gen[15] = False
    sh_parameters_gen[16] = "gen"
    sh_parameters_gen[17] = gen_num_beams
    sh_parameters_gen[18] = alpha
    return f"sh run_hf.sh {' '.join([str(v) for v in sh_parameters_gen.values()])}"

def build_step1_cmd(gen_model_path,alpha):
    sh_parameters_gen[1] = gen_model_path
    sh_parameters_gen[2] = gen_output_dir  # inference only
    sh_parameters_gen[4] = gen_train_src_toy_file
    sh_parameters_gen[10] = int(time.time())
    sh_parameters_gen[12] = ver_train_samples_per_iter
    sh_parameters_gen[13] = True
    sh_parameters_gen[14] =  ver_train_src_file
    sh_parameters_gen[15] = False
    sh_parameters_gen[16] = "ver"
    sh_parameters_gen[18] = alpha
    return f"sh run_hf.sh {' '.join([str(v) for v in sh_parameters_gen.values()])}"

def build_step4_cmd(gen_model_path, gen_save_prefix,alpha):
    sh_parameters_gen[1] = gen_model_path
    sh_parameters_gen[2] = os.path.join(gen_output_dir, gen_save_prefix)
    sh_parameters_gen[4] = gen_train_iter_file
    sh_parameters_gen[10] = int(time.time())
    sh_parameters_gen[12] = gen_train_samples_per_iter
    sh_parameters_gen[13] = False
    sh_parameters_gen[14] = gen_train_src_file
    sh_parameters_gen[15] = True
    sh_parameters_gen[16] = "gen"
    sh_parameters_gen[18] = alpha
    return f"sh run_hf.sh {' '.join([str(v) for v in sh_parameters_gen.values()])}"

gen_save_prefix, gen_model_path, ver_save_prefix, ver_model_path = None, None, None, None 
if __name__ == "__main__":
    for i in range(1, max_iter+1):
        gen_load_prefix, gen_save_prefix  = f"gen_iter_{i - 1}", f"gen_iter_{i}" 
        gen_model_path = initial_gen_path if i == 1 else os.path.join(gen_output_dir, gen_load_prefix)

        ## 0. Generaotr do self-sampling (SS-corpus) --> "gen_train_iter_unlabeled.jsonl" 
        step0_cmd = build_step0_cmd(gen_model_path,gan_alpha)
        # print_cmd_launch(step0_cmd, i, "STEP 0"); run_command(step0_cmd)

        ## 1. Gen create GAN-VER-corpus  --> "ver_train_iter_unlabled.jsonl"
        step1_cmd = build_step1_cmd(gen_model_path,gan_alpha)
        print_cmd_launch(step1_cmd, i, "STEP 1"); run_command(step1_cmd)

        ## 2. NLI label GAN Ver-Train-corpus  --> "ver_train_iter.jsonl"
        cmd_nli = f"sh run_nli_es.sh"  ### Xinyu: Adjust its inference bsz
        #print_cmd_launch(cmd_nli, i, "STEP 2"); run_command(cmd_nli)

        ## 3. Train Ver on GAN Ver-train-coprus & Label SS-corpus  --> "gen_train_iter.jsonl"
        ver_load_prefix, ver_save_prefix  = f"ver_iter_{i - 1}", f"ver_iter_{i}" 
        ver_model_path = initial_ver_path if i == 1  else os.path.join(ver_output_dir, ver_load_prefix)
        ver_save_path = os.path.join(ver_output_dir, ver_save_prefix)
        cmd_ver = f"sh run_ver_es.sh {ver_model_path} {ver_save_path}"
        print_cmd_launch(cmd_ver, i, "STEP 3"); run_command(cmd_ver)


        ## 4. Train Gen on labeled SS-corpus & create SS-corpus for next iteration  --> "gen_train_iter_unlabled.jsonl"
        step4_cmd = build_step4_cmd(gen_model_path, gen_save_prefix,gan_alpha) # notice we will not do prediction here because file format misalgins
        print_cmd_launch(step4_cmd, i, "STEP 4"); run_command(step4_cmd)



