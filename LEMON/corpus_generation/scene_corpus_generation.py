import json
import sys
import copy
from itertools import combinations, permutations
from random import choice, choices, shuffle
import math
import argparse
from multiprocessing import Pool
import multiprocessing
from collections import Counter
from functools import reduce
from math import gcd
from random import sample

parser = argparse.ArgumentParser()
parser.add_argument("--max_number", type=int, default=100000, help="max number each dataset.")
parser.add_argument("--corpus_file", type=str, default='../corpus/pretraining_corpus_scene.txt', help="corpus file")
args = parser.parse_args()

fw = open(args.corpus_file, 'w')

def lcm(numbers):
  return reduce((lambda x, y: int(x * y / gcd(x, y))), numbers)

def obtain_action_weight(actions):
    temp = Counter([item.split()[0] for item in actions])
    lcm_value = lcm(temp.values())
    temp = {item:int(lcm_value / temp[item]) for item in temp}
    action_weight = [temp[item.strip().split()[0]] for item in actions]
    return action_weight

def random_sampling(candidate_list, n, weights=None):

    result_list = []
    for _ in range(n):
        result = choices(candidate_list, k=1, weights=weights)[0]
        result_list.append(result)
    return result_list

def postpreprocess_scene(states):
    states = ' | '.join(states.split())
    return states

def scene_executor(slots, actions):
    slots = copy.deepcopy(slots)
    for action in actions:
        splits = action.split()
        if splits[0] == 'appear_person':
            if slots[int(splits[1])-1].split(':')[1] == '__':
                slots[int(splits[1])-1] = '{}:{}_'.format(slots[int(splits[1])-1].split(':')[0], splits[2])
            else:
                return 'Failed: already has a person here'
        elif splits[0] == 'appear_hat':
            if slots[int(splits[1])-1].split(':')[1][0] == '_':
                return 'Failed: No person here'
            else:
                if slots[int(splits[1])-1].split(':')[1][1] == '_':
                    slots[int(splits[1])-1] = slots[int(splits[1])-1][:-1] + splits[2]
                else:
                    return 'Failed: already has a hat here'
        elif splits[0] == 'remove_person':
            if slots[int(splits[1])-1].split(':')[1][1] != '_':
                return 'Failed: please remove hat in this position first.'
            else:
                if slots[int(splits[1])-1].split(':')[1][0] == '_':
                    return 'Failed: no person requires to remove here.'
                else:
                    slots[int(splits[1])-1] = '{}:__'.format(slots[int(splits[1])-1].split(':')[0])
        elif splits[0] == 'remove_hat':
            if slots[int(splits[1])-1].split(':')[1][0] == '_':
                return 'Failed: no person here.'
            else:
                if slots[int(splits[1])-1].split(':')[1][1] == '_':
                    return 'Failed: no hat here.'
                else:
                    slots[int(splits[1])-1] = slots[int(splits[1])-1][:-1] + '_'
        else:
            return 'Failed: No such function:{}'.format(splits[0])
            
    return slots

def scene_state_generator():
    state_element = ['b', 'g', 'o', 'p', 'r', 'y'] * 2
    all_states = list(permutations(state_element, 2))
    all_states = list(set([''.join(item) for item in all_states if not (item[0]=='_' and item[1]!='_')]))
    content = ['{}'.format(item) for i,item in enumerate(random_sampling(all_states, 2))]
    position = sample(list(range(1,11)), 2)
    state_dict = dict(zip([str(item) for item in position], content))
    for key in list(range(1,11)):
        if str(key) not in state_dict:
            state_dict[str(key)] = '__'
    states = ['{}:{}'.format(key, value) for key, value in state_dict.items()]
    states.sort(key=lambda x:int(x.split(':')[0]))
    # states = ' '.join(states)
    return states





def obtain_valid_actions_scene(states):
    action_list = []
    states = [item.strip().split(':')[1] for item in states]
    for i in range(len(states)):
        if states[i][0] == '_':
            action_list.extend(['appear_person {} {}'.format(str(i+1), item) for item in ['b', 'g', 'o', 'p', 'r', 'y']])
        else:
            if states[i][1] == '_':
                action_list.extend(['appear_hat {} {}'.format(str(i+1), item) for item in ['b', 'g', 'o', 'p', 'r', 'y']])
                action_list.append('remove_person {}'.format(str(i+1)))
            else:
                action_list.append('remove_hat {}'.format(str(i+1)))
        
    return list(set(action_list))


def scene_corpus_generation(inputs):

    total_number, action_number_range = inputs

    state_element = ['b', 'g', 'o', 'p', 'r', 'y', '_'] * 2
    all_states = list(permutations(state_element, 2))
    all_states = list(set([''.join(item) for item in all_states if not (item[0]=='_' and item[1]!='_')]))
    all_states_weight = [6 if '_' in item else 1 for item in all_states]
    index = all_states.index('__')
    all_states_weight[index] = 64
    all_actions = ['appear_person {} {}'.format(str(i+1),j) for i in range(10) for j in ['b', 'g', 'o', 'p', 'r', 'y']]
    all_actions += ['appear_hat {} {}'.format(str(i+1),j) for i in range(10) for j in ['b', 'g', 'o', 'p', 'r', 'y']]
    all_actions += ['remove_person {}'.format(str(i+1)) for i in range(10)]
    all_actions += ['remove_hat {}'.format(str(i+1)) for i in range(10)]

    count = 0
    print('Begin generating scene corpus.')
    while True:
        # prev_states = ['{}:{}'.format(str(i+1), item) for i,item in enumerate(random_sampling(all_states, 10, weights=all_states_weight))]
        prev_states = scene_state_generator()
        states_this_step = prev_states
        index = 0
        action_list = []
        step_this_case = choice(action_number_range)
        while index < step_this_case:
            all_valid_actions = obtain_valid_actions_scene(states_this_step)
            action_weight = obtain_action_weight(all_valid_actions)
            action = random_sampling(all_valid_actions, 1, weights=action_weight)
            states_this_step = scene_executor(states_this_step, action)
            assert isinstance(states_this_step, list)
            action_list.extend(action)
            index += 1
        curr_states = states_this_step

        prev_states = postpreprocess_scene(' '.join(prev_states))
        actions = ' '.join(action_list)
        curr_states = postpreprocess_scene(' '.join(curr_states))
        item_row = '\t'.join([actions, prev_states, curr_states])
        fw.write(item_row)
        fw.write('\n')
        count += 1
        if count % 10000 == 0:
            print('Finish generating {} cases'.format(count))
        if count >= total_number:
            break

if __name__ == '__main__':

    total_number_list = [int(args.max_number * 0.3), int(args.max_number * 0.4), int(args.max_number * 0.2), int(args.max_number * 0.1)]
    action_number_range_list = [list(range(1,6)), list(range(6,11)), list(range(11,16)), list(range(16,21))]

    cores = multiprocessing.cpu_count()
    print("Using {} cores".format(cores))
    pool = Pool(cores)
    for total_number, action_number_range in zip(total_number_list, action_number_range_list):
        res = pool.map(scene_corpus_generation, zip([int(total_number // cores) * cores], [action_number_range]*cores))


    # scene_corpus_generation(int(args.max_number * 0.3), list(range(1,6)))
    # scene_corpus_generation(int(args.max_number * 0.4), list(range(6,11)))
    # scene_corpus_generation(int(args.max_number * 0.2), list(range(11,16)))
    # scene_corpus_generation(int(args.max_number * 0.1), list(range(16,21)))
