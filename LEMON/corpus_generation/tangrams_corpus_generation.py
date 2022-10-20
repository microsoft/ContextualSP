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

# from corpus_generation.scene_corpus_generation import postpreprocess_scene

parser = argparse.ArgumentParser()
parser.add_argument("--max_number", type=int, default=100000, help="max number each dataset.")
parser.add_argument("--corpus_file", type=str, default='../corpus/pretraining_corpus_tangrams.txt', help="corpus file")
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

def tangrams_shape_to_letter(shape):
    return chr(int(shape) + 65)

def postpreprocess_tangrams(states):
    states = [item for item in states.split() if item.strip().split(':')[1]!='_']
    states = ['{}:{}'.format(str(i+1), tangrams_shape_to_letter(elem)) for i, elem in enumerate([item.split(':')[1] for item in states])]
    states = ['{}:{}'.format(str(i+1), elem) for i, elem in enumerate([item.split(':')[1] for item in states] + ['_'] * (5-len(states)))]
    states = ' | '.join(states)

    return states



def random_sampling(candidate_list, n, weights=None):

    result_list = []
    for _ in range(n):
        result = choices(candidate_list, k=1, weights=weights)[0]
        result_list.append(result)
    return result_list

def tangrams_letter_to_shape(letter):
    return str(ord(letter) - 65)

def tangrams_executor(slots, actions):
    content = [item.split(':')[1] for item in slots]
    content = [item for item in content if item != '_']
    for action in actions:
        splits = action.split()
        if splits[0] == 'insert':
            if len(content) >= 5:
                return 'Failed: sequence is too long.'
            else:
                if int(splits[1]) > len(content) + 1:
                    return 'Failed: index greater than sequence length'
                else:
                    if tangrams_letter_to_shape(splits[2]) in content:
                        return 'Failed: elegram already in the sequence'
                    else:
                        content.insert(int(splits[1])-1, tangrams_letter_to_shape(splits[2]))
        elif splits[0] == 'remove':
            if len(content) <= 1:
                return 'Failed: sequence is too short.'
            else:
                if int(splits[1]) > len(content):
                    return 'Failed: index greater than sequence length'
                else:
                    del content[int(splits[1])-1]

    slots = ['{}:{}'.format(str(i+1), item) for i, item in enumerate(content)]
    if len(slots) < 5:
        slots = ['{}:{}'.format(str(i+1), elem) for i, elem in enumerate([item.split(':')[1] for item in slots] + ['_'] * (5-len(slots)))]
    return slots

def tangrams_state_generator():
    all_states = list(set(list(permutations(list(range(5)), 5))))
    states = ['{}:{}'.format(str(i+1), item) for i,item in enumerate(random_sampling(all_states, 1)[0])]
    # states = ' '.join(states)
    return states

def obtain_valid_actions_tangrams(states):
    action_list = []
    states = [item.strip().split(':')[1] for item in states]
    total_len = len([item for item in states if item != '_'])
    if total_len < 5:
        action_list.extend(['insert {} {}'.format(str(i+1), j) for i in range(total_len+1) for j in ['A','B','C','D','E'] if tangrams_letter_to_shape(j) not in states])
    if total_len > 1:
        action_list.extend(['remove {}'.format(str(i+1)) for i in range(total_len)])
    
    return list(set(action_list))

def tangrams_corpus_generation(inputs):

    total_number, action_number_range = inputs

    all_states = list(set(list(permutations(list(range(5)), 5)) + list(permutations(list(range(5)), 4)) + list(permutations(list(range(5)), 3)) + list(permutations(list(range(5)), 2)) + list(permutations(list(range(5)), 1))))

    all_actions = ['insert {} {}'.format(str(i+1), j) for i in range(5) for j in ['A', 'B', 'C', 'D', 'E']]
    # all_actions = ['insert {} {}'.format(str(i+1), j) for i in range(5) for j in [0,1,2,3,4]]
    all_actions += ['remove {}'.format(str(i+1)) for i in range(5)]

    count = 0
    print('Begin generating tangrams corpus.')
    while True:
        # prev_states = ['{}:{}'.format(str(i+1), item) for i,item in enumerate(random_sampling(all_states, 1)[0])]
        prev_states = tangrams_state_generator()
        if len(prev_states) < 5:
            prev_states = ['{}:{}'.format(str(i+1), elem) for i, elem in enumerate([item.split(':')[1] for item in prev_states] + ['_'] * (5-len(prev_states)))]

        states_this_step = prev_states
        index = 0
        action_list = []
        step_this_case = choice(action_number_range)
        while index < step_this_case:
            all_valid_actions = obtain_valid_actions_tangrams(states_this_step)
            action_weight = obtain_action_weight(all_valid_actions)
            action = random_sampling(all_valid_actions, 1, weights=action_weight)
            states_this_step = tangrams_executor(states_this_step, action)
            assert isinstance(states_this_step, list)
            action_list.extend(action)
            index += 1
        curr_states = states_this_step

        prev_states = postpreprocess_tangrams(' '.join(prev_states))
        actions = ' '.join(action_list)
        curr_states = postpreprocess_tangrams(' '.join(curr_states))
        item_row = '\t'.join([actions, prev_states, curr_states])
        fw.write(item_row)
        fw.write('\n')
        count += 1
        if count % 10000 == 0:
            print('Finish generating {} cases'.format(count))
        if count >= total_number:
            break


if __name__ == '__main__':

    total_number_list = [int(args.max_number * 0.35), int(args.max_number * 0.4), int(args.max_number * 0.15), int(args.max_number * 0.1)]
    action_number_range_list = [list(range(1,6)), list(range(6,11)), list(range(11,16)), list(range(16,21))]

    cores = multiprocessing.cpu_count()
    print("Using {} cores".format(cores))
    pool = Pool(cores)
    for total_number, action_number_range in zip(total_number_list, action_number_range_list):
        res = pool.map(tangrams_corpus_generation, zip([int(total_number // cores) * cores], [action_number_range]*cores))


    # tangrams_corpus_generation(int(args.max_number * 0.35), list(range(1,6)))
    # tangrams_corpus_generation(int(args.max_number * 0.4), list(range(6,11)))
    # tangrams_corpus_generation(int(args.max_number * 0.15), list(range(11,16)))
    # tangrams_corpus_generation(int(args.max_number * 0.1), list(range(16,21)))