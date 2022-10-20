import os
import json
import argparse
import subprocess
from tqdm import tqdm
from step1_schema_linking import read_database_schema
from train import run_command

def running_process(generate_path):
    # cmd = f"python -m multiprocessing_bpe_encoder \
    #         --encoder-json ./models/spider_sl/encoder.json \
    #         --vocab-bpe ./models/spider_sl/vocab.bpe \
    #         --inputs {generate_path}/train.src \
    #         --outputs {generate_path}/train.bpe.src \
    #         --workers 1 \
    #         --keep-empty"
    # run_command(cmd)
    #
    # cmd = f"python -m multiprocessing_bpe_encoder \
    #         --encoder-json ./models/spider_sl/encoder.json \
    #         --vocab-bpe ./models/spider_sl/vocab.bpe \
    #         --inputs {generate_path}/train.tgt \
    #         --outputs {generate_path}/train.bpe.tgt \
    #         --workers 1 \
    #         --keep-empty"
    # run_command(cmd)

    cmd = f"python -m multiprocessing_bpe_encoder \
            --encoder-json ./models/spider_sl/encoder.json \
            --vocab-bpe ./models/spider_sl/vocab.bpe \
            --inputs {generate_path}/dev.src \
            --outputs {generate_path}/dev.bpe.src \
            --workers 1 \
            --keep-empty"
    run_command(cmd)

    cmd = f"python -m multiprocessing_bpe_encoder \
            --encoder-json ./models/spider_sl/encoder.json \
            --vocab-bpe ./models/spider_sl/vocab.bpe \
            --inputs {generate_path}/dev.tgt \
            --outputs {generate_path}/dev.bpe.tgt \
            --workers 1 \
            --keep-empty"
    run_command(cmd)

    # cmd = f'fairseq-preprocess --source-lang "src" --target-lang "tgt" \
    #     --trainpref {generate_path}/train.bpe \
    #     --validpref {generate_path}/dev.bpe \
    #     --destdir {generate_path}/bin \
    #     --workers 2 \
    #     --srcdict ./models/spider_sl/dict.src.txt \
    #     --tgtdict ./models/spider_sl/dict.tgt.txt '

    cmd = f'fairseq-preprocess --source-lang "src" --target-lang "tgt" \
        --validpref {generate_path}/dev.bpe \
        --destdir {generate_path}/bin \
        --workers 2 \
        --srcdict ./models/spider_sl/dict.src.txt \
        --tgtdict ./models/spider_sl/dict.tgt.txt '


    subprocess.Popen(
        cmd, universal_newlines=True, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

def build_schema_linking_data(schema, question, item, turn_id, linking_type):
    source_sequence_list, target_sequence_list = [], []

    # column description
    column_names = []
    for i, (t, c) in enumerate(zip(schema['column_types'], schema['column_names_original'])):
        if c[0] == -1:
            column_names.append("{0} {1}".format(t, c[1].lower()))
        else:
            column_with_alias = "{0}@{1}".format(schema['table_names_original'][c[0]].lower(), c[1].lower())
            tag_list = []
            if column_with_alias in item['interaction'][turn_id]['exact_match']:
                tag_list.append('EM')
            elif column_with_alias in item['interaction'][turn_id]['partial_match']:
                tag_list.append('PA')
            if column_with_alias in item['interaction'][turn_id]['value_match']:
                tag_list.append('VC')

            # primary-foreign key
            if i in schema['primary_keys']:
                tag_list.append('RK')
            elif i in schema['foreign_keys_col']:
                tag_list.append('FO')

            if tag_list != []:
                column_names.append("{0} {1} {2}".format(' '.join(tag_list), t, column_with_alias))
            else:
                column_names.append("{0} {1}".format(t, column_with_alias))

    # table description
    table_names = []
    for t in schema['table_names_original']:
        tag_list = []
        if t in item['interaction'][turn_id]['exact_match']:
            tag_list.append('EM')
        elif t in item['interaction'][turn_id]['partial_match']:
            tag_list.append('PA')
        if '_nosl' in linking_type or 'not' in linking_type:
            tag_list = []

        if tag_list != []:
            table_names.append("{0} {1}".format(' '.join(tag_list), t.lower()))
        else:
            table_names.append("{0}".format(t.lower()))

    table_names = ' | '.join(table_names)
    column_names = ' | '.join(column_names)
    for structure_schema_list in schema['permutations'][:10]:
        structure_schema_str = ' | '.join(structure_schema_list)

    source_sequence = f"<C> {column_names} | <T> {table_names} | <S> {structure_schema_str} | <Q> {question.lower()}"
    target_sequence = item['interaction'][turn_id]['sql'].lower()

    source_sequence_list.append(source_sequence)
    target_sequence_list.append(target_sequence)

    return source_sequence_list, target_sequence_list

def extract_input_and_output(example_lines, linking_type):
    inputs = []
    outputs = []
    database_schema_filename = './data/spider/tables.json'


    schema_tokens, column_names, database_schemas = read_database_schema(database_schema_filename)

    for item in tqdm(example_lines):
        question = item['interaction'][0]['utterance']
        schema = database_schemas[item['database_id']]
        source_sequence, target_sequence = build_schema_linking_data(schema=schema,
                                                                     question=question,
                                                                     item=item,
                                                                     turn_id=0,
                                                                     linking_type=linking_type)
        outputs.extend(target_sequence)
        inputs.extend(source_sequence)


    assert len(inputs) == len(outputs)
    return inputs, outputs


def read_dataflow_dataset(file_path, out_folder, session, linking_type):
    train_out_path = os.path.join(out_folder, session)
    train_src_writer = open(train_out_path + ".src", "w", encoding="utf8")
    train_tgt_writer = open(train_out_path + ".tgt", "w", encoding="utf8")

    with open(file_path, "r", encoding='utf-8') as data_file:
        lines = json.load(data_file)
        data_input, data_output = extract_input_and_output(lines, linking_type)
        train_src_writer.write("\n".join(data_input))
        train_tgt_writer.write("\n".join(data_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sl_dataset_path", default='./data/spider_schema_linking_tag')
    parser.add_argument("--output_path", default='./dataset_post/spider_sl')
    parser.add_argument("--linking_type", default='default')
    args = parser.parse_args()

    # for session in ["train", "dev"]:
    for session in ["dev"]:
        file_path = os.path.join(args.sl_dataset_path, "{}.json".format(session))
        out_folder = args.output_path
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        read_dataflow_dataset(file_path, out_folder, session, args.linking_type)
    running_process(args.output_path)