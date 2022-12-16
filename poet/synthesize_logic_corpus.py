from z3 import *
import random
from random import shuffle
from itertools import combinations, product
from typing import List, Tuple
from functools import partial
from tqdm import tqdm
import os


solver = Solver()
vars_all_candidates = [chr(i) for i in list(range(97, 122))]
for symbol in vars_all_candidates:
    # init all variables
    exec("{0} = Bool('{0}')".format(symbol))


def sample_single_logic(var_inputs: Tuple):
    var_1, var_2 = var_inputs
    # given two vars, sample a logic to represent these
    # decide order of two vars
    logic_var_1 = var_1
    logic_var_2 = var_2
    if random.random() > 0.5:
        var_1 = "not {}".format(var_1)
        logic_var_1 = "Not({})".format(logic_var_1)

    if random.random() > 0.5:
        var_2 = "not {}".format(var_2)
        logic_var_2 = "Not({})".format(logic_var_2)

    # implies of two logic var
    if random.random() > 0.5:
        var_1, var_2 = var_2, var_1
        logic_var_1, logic_var_2 = logic_var_2, logic_var_1

    text = "( {} -> {} ) ;".format(var_1, var_2)
    logic = "Implies({}, {})".format(logic_var_1, logic_var_2)
    return logic, text


def sample_simple_hypo(var_candidates: List):
    # sample 1 or 2
    sample_num = 1 if random.random() < 0.75 else 2
    var_combinations = list(combinations(var_candidates, 2))
    shuffle(var_combinations)
    if sample_num == 1 or len(var_combinations) == 1:
        # select the first one as the var_candidates
        var_1, var_2 = var_combinations[0]
        return sample_single_logic((var_1, var_2))

    sample_predicate = "And" if random.random() < 0.75 else "Or"
    var_1, var_2 = var_combinations[0]
    logic_1 = sample_single_logic((var_1, var_2))
    var_1, var_2 = var_combinations[1]
    logic_2 = sample_single_logic((var_1, var_2))
    # build both
    logic_part = sample_predicate + "({}, {})".format(logic_1[0], logic_2[0])
    if sample_predicate == "And":
        text_part = logic_1[1] + " " + logic_2[1]
    else:
        text_part = logic_1[1] + " or " + logic_2[1]
    return logic_part, text_part


def validate_statements(fact: List, hypo_linear: str = None):
    solver.reset()
    # construct the overall statement
    if hypo_linear is None:
        fact_linear = ", ".join(fact)
        logic_state = "And({})".format(fact_linear)
        exec("solver.add(" + logic_state + ")")
        result = solver.check()
        if result.r == 1:
            return True
        else:
            return False
    else:
        fact_linear = ", ".join(fact)
        logic_state = "Not(Implies(And({0}), {1}))".format(fact_linear, hypo_linear)
        exec("solver.add(" + logic_state + ")")
        result = solver.check()
        if result.r == -1:
            return True
        else:
            return False


def sample_example():
    vars_candidates = [chr(i) for i in list(range(97, 122))]
    shuffle(vars_candidates)
    # take the first 3-4 to construct the complete logic rules
    # total_var_count = random.randint(4, 9)
    total_var_count = 4
    use_var_count = min(random.randint(2, total_var_count - 2), 4)
    used_vars = vars_candidates[:use_var_count]
    # take the 4-10 to construct negative examples
    unused_vars = vars_candidates[use_var_count: total_var_count]
    # for each pair of vars, firstly we construct var pairs
    used_combines = list(combinations(used_vars, 2))
    # we should verify that the logic is valid
    context_logic_is_valid = False
    logic_facts = []
    text_facts = []

    while not context_logic_is_valid:
        shuffle(used_combines)
        # take 3~5 from them
        count_combines = random.randint(3, 6)
        used_combines = used_combines[: count_combines]
        logic_facts = list(map(sample_single_logic, used_combines))
        # if the logic in context is valid, break
        logic_facts, text_facts = zip(*logic_facts)
        context_logic_is_valid = validate_statements(logic_facts)

    logic_facts = list(logic_facts)
    text_facts = list(text_facts)

    # add some other facts
    unused_combines = list(combinations(unused_vars, 2))
    shuffle(unused_combines)
    unused_text_facts = list(map(sample_single_logic, unused_combines[: random.randint(5, 15)]))
    _, unused_text_facts = zip(*unused_text_facts)
    unused_text_facts = list(unused_text_facts)

    # add some used var and unused var facts which cannot affect the logic
    add_negative_count = random.randint(1, 2)
    all_combines = list(product(vars_candidates[:use_var_count], vars_candidates[use_var_count:total_var_count]))
    shuffle(all_combines)
    while add_negative_count > 0:
        # try to add one
        temp_logic_facts = logic_facts[:]
        cur_logic, cur_text = sample_single_logic(all_combines[-add_negative_count])
        temp_logic_facts.append(cur_logic)
        if validate_statements(temp_logic_facts):
            # update logic facts and textual facts
            logic_facts.append(cur_logic)
            text_facts.append(cur_text)
        add_negative_count -= 1

    # sample logic hypo
    logic_hypo, text_hypo = None, ""
    while logic_hypo is None:
        logic_hypo, text_hypo = sample_simple_hypo(used_vars)
        if logic_hypo in logic_facts:
            # too trivial to verify
            logic_hypo = None

    # give an answer
    text_facts = list(text_facts + unused_text_facts)
    shuffle(text_facts)

    text_final = text_hypo + " [SEP] " + " ".join(text_facts)
    answer_final = "1" if validate_statements(logic_facts, logic_hypo) else "0"

    return text_final, answer_final


def convert_logical_data(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_src_f = open(os.path.join(output_dir, "train.raw.input0"), "w", encoding="utf8")
    train_tgt_f = open(os.path.join(output_dir, "train.label"), "w", encoding="utf8")
    dev_src_f = open(os.path.join(output_dir, "dev.raw.input0"), "w", encoding="utf8")
    dev_tgt_f = open(os.path.join(output_dir, "dev.label"), "w", encoding="utf8")

    for _ in tqdm(range(100000)):
        input_line, output_line = sample_example()
        train_src_f.write(input_line + "\n")
        train_tgt_f.write(output_line + "\n")

    for _ in tqdm(range(2000)):
        input_line, output_line = sample_example()
        dev_src_f.write(input_line + "\n")
        dev_tgt_f.write(output_line + "\n")

    train_src_f.close()
    train_tgt_f.close()
    dev_src_f.close()
    dev_tgt_f.close()


if __name__ == '__main__':
    convert_logical_data("pretrain_logic")