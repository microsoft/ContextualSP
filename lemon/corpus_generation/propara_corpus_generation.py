


# def propara_executor(state, action):

import jsonlines
from tqdm import tqdm
from random import choices
import argparse
import multiprocessing
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_prefix", type=str, default='propara', help="dataset prefix")
# parser.add_argument("--max_number", type=int, default=10000, help="max number each dataset.")
parser.add_argument("--corpus_file", type=str, default='../corpus/pretraining_corpus_propara.txt', help="corpus file")
args = parser.parse_args()

fw = open(args.corpus_file, 'w')


def random_sampling(candidate_list, n, weights=None):

    result_list = []
    for _ in range(n):
        result = choices(candidate_list, k=1, weights=weights)[0]
        result_list.append(result)
    return result_list

def propara_state_generator(candidate_list):
    # random get a state
    item = random_sampling(candidate_list, 1)[0]
    participants = item['participants']
    states = random_sampling(item['states_token'], len(participants))
    states = {'participants':participants, 'states':states}
    return states

def propara_action_generator(states, states_tokens):
    # random get an action
    action = ''
    func_list = ['Move', 'Create', 'Destroy']
    while True:
        try:
            func = random_sampling(func_list,1)[0]
            if func == 'Create':
                available_participants = [p for p, s in zip(states['participants'],states['states']) if s == '-']
                p = random_sampling(available_participants, 1)[0]
                x = random_sampling(states_tokens, 1)[0]
                if x not in ['-', '?']:
                    action = {'func': func, 'participant':p, 'para1':x}
                else:
                    action = {'func': func, 'participant':p}

            elif func == 'Destroy':
                available_participants = [p for p, s in zip(states['participants'],states['states']) if s != '-']
                p = random_sampling(available_participants, 1)[0]
                action = {'func': func, 'participant':p}
                
            elif func == 'Move':
                available_participants_states = [(p,s) for p, s in zip(states['participants'],states['states']) if s != '-']
                p, x1 = random_sampling(available_participants_states, 1)[0]
                x2 = random_sampling([item for item in states_tokens if item != x1 and item != '-'], 1)[0]
                action = {'func': func, 'participant':p, 'para1':x1, 'para2':x2}
            
            break

        except:
            continue

    return action

def propara_exeutor(states, action):
        
    result = dict()
    if action['func'] == 'Create':
        if 'para1' not in action.keys():
            result['participants'] = states['participants']
            result['states'] = ['?' if p==action['participant'] else s for p, s in zip(states['participants'], states['states'])]
        else:
            result['participants'] = states['participants']
            result['states'] = [action['para1'] if p==action['participant'] else s for p, s in zip(states['participants'], states['states'])]
    elif action['func'] == 'Destroy':
        result['participants'] = states['participants']
        result['states'] = ['-' if p==action['participant'] else s for p, s in zip(states['participants'], states['states'])]
    elif action['func'] == 'Move':
        result['participants'] = states['participants']
        result['states'] = [action['para2'] if p==action['participant'] and s==action['para1'] else s for p, s in zip(states['participants'], states['states'])]

    return result

def states_linearize_ori(states):    
    return 'col : ' + ' | '.join(states['participants']) + ' ' + 'state : ' + ' | '.join(states['states'])

def states_linearize_tgt(states):
    return 'state : ' + ' | '.join(states['states'])


def action_linearize(action):
    if action['func'] == 'Create':
        if 'para1' in action.keys():
            result = ' '.join([action['func'], action['participant'], action['para1']])
        else:
            result = ' '.join([action['func'], action['participant']])
    elif action['func'] == 'Destroy':
        result = ' '.join([action['func'], action['participant']])
    elif action['func'] == 'Move':
        result = ' '.join([action['func'], action['participant'], 'from', action['para1'], 'to', action['para2']])

    return result

def corpus_generation(inputs):

    candidate_list, max_step, total_number = inputs

    count = 0
    while True:
        states = propara_state_generator(candidate_list)
        action_list = []
        for _ in range(20):
            action = propara_action_generator(states, states_tokens)
            if action['participant'] not in [item['participant'] for item in action_list]:
                action_list.append(action)
                if len(action_list) >= max_step:
                    break
        
        if len(action_list) == max_step:
            states_temp = states
            for action in action_list:
                states_temp = propara_exeutor(states_temp, action)
            final_states = states_temp
            initial_states = states_linearize_ori(states)
            final_states = states_linearize_tgt(final_states)
            final_action = ' , '.join([action_linearize(item) for item in action_list]).lower()
            item_row = '\t'.join([final_action.strip(), initial_states, final_states]).lower()
            fw.write(item_row)
            fw.write('\n')
            count += 1
            if count % 10000 == 0:
                print('Finish generating {} cases'.format(count))
            if count >= total_number:
                break



if __name__ == '__main__':

    if args.dataset_prefix == 'propara':
        data_lines = list(jsonlines.open('./grids.v1.train.json', 'r'))
        candidate_list = []
        for index in tqdm(range(len(data_lines))):
            line = data_lines[index]
            id = line['para_id']
            participants = line['participants']
            states = line['states']
            states_tokens = [states[i][j] for i in range(len(states)) for j in range(len(states[0]))]
            candidate = {'participants':participants, 'states_token':states_tokens}
            candidate_list.append(candidate)


        cores = multiprocessing.cpu_count()
        print("Using {} cores".format(cores))
        pool = Pool(cores)

        max_number_list = [200000,300000,300000,300000,150000,75000,25000,10000]     # 100W

        
        for i in range(1,9):
            res = pool.map(corpus_generation, zip([candidate_list]*cores, [i]*cores, [int(max_number_list[i-1] // cores)]*cores))

        pool.close()
        pool.join()


