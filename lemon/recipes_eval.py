import json
import argparse
from pydoc import doc
import collections
import os


def get_col_states(input_str):
    col_and_state = input_str.replace('state : ', '').split(' | ')
    return col_and_state


def get_col_states_start(input_str):
    col_and_state = input_str.split(' states : ')
    cols = col_and_state[0].replace('col : ', '').split(' | ')
    states = col_and_state[1].split(' | ')
    states[-1] = states[-1].split(' SEP ')[0]
    return cols, states


def get_action(location_before, location_after):
    location_before = location_before.replace("states : ", '')
    location_after = location_after.replace("states : ", "")
    if location_before == location_after:
        return "NONE",location_before, location_after
    if location_before == '-' and location_after != '-':
        return "CREATE",location_before, location_after
    if location_after == '-' and location_before != '-':
        return "DESTROY",location_before, location_after
    if location_before != '-' and location_after != '-':
        return "MOVE",location_before, location_after


def process(id_path, generate_valid_path, dummy_path, if_answer=False):
    target_idx = 2 if if_answer else 1
    
    error_num = 0
    id_file = open(id_path, 'r', encoding='utf8')
    pre = open(generate_valid_path, 'r', encoding='utf8')
    out = open(dummy_path, 'w', encoding='utf8')

    linenum_to_colandstate = {}
    pre_lines = pre.readlines()[1:]

    for line in pre_lines:
        elements = line.rstrip().split('\t')
        line_id = int(elements[-1])
        col_and_state = elements
        linenum_to_colandstate[line_id] = col_and_state

    current_case = -1
    pre_states = {}

    id_lines = id_file.readlines()

    step_num = 0
    action_matrix = collections.OrderedDict()
    for line_id, case_id in enumerate(id_lines):

        case_id, step_id = case_id.rstrip().split('-')  # '4-1' -> [4, 1]

        if case_id != current_case:
            for key in action_matrix.keys():
                for step_idx in range(step_num):
                    try:
                        line_out = str(current_case) + '\t' + str(step_idx + 1) + '\t' + key + '\t' + action_matrix[key][
                            step_idx][0] + '\t' + action_matrix[key][step_idx][1] + '\t' + action_matrix[key][step_idx][2] + '\t'
                        out.write(line_out + '\n')
                    except:
                        line_out = str(current_case) + '\t' + str(step_idx + 1) + '\t' + key + '\t' + 'NONE' + '\t' + '-' + '\t' + '-' + '\t'
                        out.write(line_out + '\n')

            action_matrix = {}
            step_num = 0

            current_case = case_id
            start_col_and_state = linenum_to_colandstate[line_id][-2]
            pre_cols, pre_states = get_col_states_start(start_col_and_state)  # get the init state
            for key in pre_cols:
                action_matrix[key] = []  # init the action matrix

        step_num += 1
        col_and_state = linenum_to_colandstate[line_id][target_idx]  # get the first state (after the first action)
        current_states = get_col_states(col_and_state) # current_states : List : ['state1', 'state2', 'state3', 'state4']

        if len(current_states) != len(pre_states):
            error_num += 1

        col_list = list(action_matrix.keys())

        for col_idx in range(len(col_list)):
            try:
                action_matrix[col_list[col_idx]].append((get_action(pre_states[col_idx], current_states[col_idx])))
            except:
                right_col = col_list[col_idx]
                pre_state = '-' if col_idx >= len(pre_states) else pre_states[col_idx]
                current_state = '-' if col_idx >= len(current_states) else current_states[col_idx]
                error_action = (get_action(pre_state, current_state))
                action_matrix[right_col].append(error_action)

        pre_states = current_states
    
    for key in action_matrix.keys():
        for step_idx in range(step_num):
            try:
                line_out = str(current_case) + '\t' + str(step_idx + 1) + '\t' + key + '\t' + action_matrix[key][
                    step_idx][0] + '\t' + action_matrix[key][step_idx][1] + '\t' + action_matrix[key][step_idx][2] + '\t'
                out.write(line_out + '\n')
            except:
                line_out = str(current_case) + '\t' + str(step_idx + 1) + '\t' + key + '\t' + 'NONE' + '\t' + '-' + '\t' + '-' + '\t'
                out.write(line_out + '\n')

    print('error_num', error_num)

def eval_recipes_stage2(prediction_file, answer_file, predict_target_file, answer_target_file):

    predict_list = open(prediction_file, 'r', encoding='utf8').readlines()
    answer_list = open(answer_file, 'r', encoding='utf8').readlines()
    predict_dict = dict()
    answer_dict = dict()
    for predict, answer in zip(predict_list, answer_list):
        predict_item = predict.strip().split('\t')
        answer_item = answer.strip().split('\t')
        assert predict_item[0] == answer_item[0]
        assert predict_item[1] == answer_item[1]
        assert predict_item[2] == answer_item[2]
        doc_id = predict_item[0]
        sentence_id = predict_item[1]
        entity = predict_item[2]
        predicted_action = predict_item[3]
        answer_action = answer_item[3]

        if (doc_id, entity) not in predict_dict:
            predict_dict[(doc_id, entity)] = []
        if predicted_action != 'NONE':
            if not (predicted_action == 'CREATE' and predict_item[5] == '?'):
                predict_dict[(doc_id, entity)].append({'step':int(sentence_id)-1, 'location': predict_item[5]})
        
        if (doc_id, entity) not in answer_dict:
            answer_dict[(doc_id, entity)] = []
        if answer_action != 'NONE':
            if not (answer_action == 'CREATE' and answer_item[5] == '?'):
                answer_dict[(doc_id, entity)].append({'step':int(sentence_id)-1, 'location': answer_item[5]})
        

    predict_json_lines = []
    for item in predict_dict:
        predict_json_lines.append({'id':int(item[0]),
                            'entity':item[1],
                            'loc_change':predict_dict[item]})

    json.dump(predict_json_lines, open(predict_target_file, 'w', encoding='utf8'), indent=4, ensure_ascii=False)

    answer_json_lines = []
    for item in answer_dict:
        answer_json_lines.append({'id':int(item[0]),
                            'entity':item[1],
                            'loc_change':answer_dict[item]})

    json.dump(answer_json_lines, open(answer_target_file, 'w', encoding='utf8'), indent=4, ensure_ascii=False)

def eval_recipes_stage3(prediction_file, answer_file):

    predict_list = json.load(open(prediction_file, 'r', encoding='utf8'))
    answer_list = json.load(open(answer_file, 'r', encoding='utf8'))

    assert len(predict_list) == len(answer_list)
    num_data = len(answer_list)
    total_pred, total_ans, total_correct = 0, 0, 0

    for idx in range(num_data):
        prediction = predict_list[idx]
        answer = answer_list[idx]

        assert prediction['id'] == answer['id'] and prediction['entity'] == answer['entity']
        pred_loc = prediction['loc_change']
        ans_loc = answer['loc_change']

        num_pred = len(pred_loc)
        num_ans = len(ans_loc)

        if num_pred == 0 or num_ans == 0:
            num_correct = 0
        else:
            num_correct = len([loc for loc in pred_loc if loc in ans_loc])

        total_pred += num_pred
        total_ans += num_ans
        total_correct += num_correct

    precision = total_correct / total_pred
    recall = total_correct / total_ans
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print(f'{num_data} instances evaluated.')
    print(f'Total predictions: {total_pred}, total answers: {total_ans}, total correct predictions: {total_correct}')
    print(f'Precision: {precision*100:.2f}, Recall: {recall*100:.2f}, F1: {f1*100:.2f}')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



def eval_all(root_path, stage):

    id_path = os.path.join('./before_pretraining_tsv/before-pretraining-1/', stage+'.id')
    generate_prediction_file = os.path.join(root_path, 'generate-'+stage+'.txt.eval')
    prediction_file = os.path.join(root_path, stage+'-predictions.tsv')
    answer_file = os.path.join(root_path, stage+'-answers.tsv')
    predict_target_file = os.path.join(root_path, stage+'_predict_loc.json')
    answer_target_file = os.path.join(root_path, stage+'_answer_loc.json')


    process(id_path, generate_prediction_file, prediction_file, False)
    process(id_path, generate_prediction_file, answer_file, True)

    eval_recipes_stage2(prediction_file, answer_file, predict_target_file, answer_target_file)
    eval_recipes_stage3(predict_target_file, answer_target_file)

if __name__ == '__main__':

    # eval_dir = 'CHECKPOINT-DIR'
    # stage = 'valid'
    # eval_all(eval_dir, stage)

    eval_dir = '/mnt/v-qshi/project/amlk8s/LEMON/models/finetune-recipes-after-pretraining-without-destroy-seed-44/checkpoint_115_7500'
    stage = 'test'
    eval_all(eval_dir, stage)