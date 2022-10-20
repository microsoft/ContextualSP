import sys
sys.path.append('../executor/')
from strongsup.rlong.executor import RLongExecutor
from strongsup.rlong.predicate import RLongPredicate
from strongsup.rlong.state import RLongAlchemyState
from itertools import permutations
from random import choices, choice, sample
import math
import argparse
import multiprocessing
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--max_number", type=int, default=100000, help="max number each dataset.")
parser.add_argument("--corpus_file", type=str, default='../corpus/pretraining_corpus_alchemy.txt', help="corpus file")
parser.add_argument("--dataset_prefix", type=str, default='alchemy', help="dataset name")
args = parser.parse_args()

fw = open(args.corpus_file, 'w')

def random_sampling(candidate_list, n, weights=None):

    result_list = []
    for _ in range(n):
        result = choices(candidate_list, k=1, weights=weights)[0]
        result_list.append(result)
    return result_list

def prepare_lf(lf):
    if isinstance(lf, str):
        lf = lf.split()
    if not all(isinstance(x, RLongPredicate) for x in lf):
        lf = [x if isinstance(x, RLongPredicate) else RLongPredicate(x)
                for x in lf]
    return lf

def postpreprocess_alchemy(states):
    return ' | '.join(states.strip().split())


def uni_executor(state, lf, dataset_prefix):
    if dataset_prefix == 'alchemy':
        state = RLongAlchemyState.from_raw_string(state)

    lf = prepare_lf(lf)
    executor = RLongExecutor(state, debug=False)

    # Direct execution
    denotation_direct = executor.execute(lf)
    # Token-by-token execution
    denotation = None
    for x in lf:
        denotation = executor.execute_predicate(x, denotation)
    assert denotation_direct == denotation

    # return denotation.world_state
    return denotation

def alchemy_state_generator():

    colors = ['b', 'g', 'o', 'p', 'r', 'y']
    num_positions = 7
    objects = []
    for i in range(num_positions):
        amt = choice([0,1,2,3,4])
        color = choice(colors)
        beaker = None
        for _ in range(amt):
            if beaker is None:
                beaker = []
            beaker.append(color)
        if beaker is None:
            beaker = '_'
        objects.append(''.join(beaker))
    
    states = ['{}:{}'.format(str(i+1), item) for i,item in enumerate(objects)]
    states = ' '.join(states)
    return states

def single_alchemy_lf_generator(states, executor, lf):

    colors = ['b', 'g', 'o', 'p', 'r', 'y']
    object_list = ['{} PColor {} index'.format(color, ind) for color in colors for ind in range(1,8)]      # 这里可能 会有问题 如果唯一的对象再制定了index 可能会累赘 先这样
    object_list += ['{} PColor'.format(color) for color in colors]
    object_list += ['all-objects {} index'.format(ind) for ind in range(1,8)]
    object_list += ['{} H1'.format(item) for item in [1,2,3,4,-1]]
    object_list += ['{} H2'.format(item) for item in [1,2,3,4,-1]]

    func = random_sampling(['APour', 'ADrain', 'AMix'], 1)[0]

    if func == 'ADrain':
        valid_objects = []
        for item in list(set(object_list)):         # shuffle because only choose one 
            try:
                result = executor(states, lf + ' ' + item, 'alchemy')
                if len(result.execution_stack[0]) == 1:
                    if str(result.execution_stack[0]).split(':')[1] != '_':
                        valid_objects.append(item)
                        break
            except:
                pass
        
        assert len(valid_objects) <= 1
        object = random_sampling(valid_objects, 1)[0]
        stack = executor(states, lf + ' ' + object, 'alchemy').execution_stack[0]
        assert len(stack) == 1
        cur_len = len(str(stack[0]).split(':')[1])
        number_list = []
        if cur_len == 4:
            number_list.extend(['X1/2', 'X1/4', '4'])
        elif cur_len == 3:
            number_list.extend(['X1/3', 'X2/3', '3'])
        elif cur_len == 2:
            number_list.extend(['X1/2', '2'])
        elif cur_len == 1:
            number_list.extend(['1'])
        number = random_sampling(number_list, 1)[0]
        if lf and func == lf.split()[-1]:
            lf += ' ' + object + ' ' + number + ' ' + '-1 H0'
        else:
            lf += ' ' + object + ' ' + number + ' ' + func
        assert executor(states, lf, 'alchemy')
    
    elif func == 'APour':
        valid_objects = []
        for item1 in list(set(object_list)):
            for item2 in list(set(object_list)):
                try:
                    result = executor(states, lf + ' ' + item1 + ' ' + item2 + ' ' + func, 'alchemy')
                    valid_objects.append((item1, item2))
                    break
                except:
                    pass
            if len(valid_objects) > 0:
                break

        assert len(valid_objects) <= 1
        object = random_sampling(valid_objects, 1)[0]
        if lf and func == lf.split()[-1]:
            lf += ' ' + object[0] + ' ' + object[1] + ' ' + '-1 H0'
        else:
            lf += ' ' + object[0] + ' ' + object[1] + ' ' + func
        assert executor(states, lf, 'alchemy')
    
    elif func == 'AMix':
        valid_objects = []
        for item in list(set(object_list)):
            try:
                result = executor(states, lf + ' ' + item + ' ' + func, 'alchemy')
                valid_objects.append(item)
                break
            except:
                pass
        
        assert len(valid_objects) <= 1
        object = random_sampling(valid_objects, 1)[0]
        lf += ' ' + object + ' ' + func
        assert executor(states, lf, 'alchemy')

    return lf
            

def lf_generator(states, executor, max_step, dataset_prefix):

    if dataset_prefix == 'alchemy':
        func = single_alchemy_lf_generator

    count = 0
    lf = ''
    for _ in range(10):
        try:
            lf = func(states, executor, lf)
        except:
            continue
        
        count += 1
        if count >= max_step:
            break
    
    return lf

def corpus_generation(inputs):

    executor, max_step, total_number, dataset_prefix = inputs

    if dataset_prefix == 'alchemy':
        state_generator = alchemy_state_generator
        state_preprocesser = postpreprocess_alchemy

    count = 0
    while True:
        states = state_generator()
        lf = lf_generator(states, executor, max_step, dataset_prefix)
        if lf.strip():
            result = executor(states, lf.strip(), dataset_prefix)
            if len(result.command_history) == max_step:
                initial_state = state_preprocesser(states)
                final_state = state_preprocesser(str(result.world_state))
                item_row = '\t'.join([lf.strip(), initial_state, final_state])
                fw.write(item_row)
                fw.write('\n')
                count += 1
                if count % 10000 == 0:
                    print('Finish generating {} cases'.format(count))
                if count >= total_number:
                    break


if __name__ == '__main__':

    cores = multiprocessing.cpu_count()
    print("Using {} cores".format(cores))
    pool = Pool(cores)

    
    for i in range(1,6):
        res = pool.map(corpus_generation, zip([uni_executor]*cores, [i]*cores, [int(args.max_number // 5 // cores)]*cores, [args.dataset_prefix]*cores))

    pool.close()
    pool.join()