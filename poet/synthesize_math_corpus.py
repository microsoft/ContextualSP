import random
from random import shuffle
import os
from tqdm import tqdm


def expand_numbers_in_text(text, delim=" ", ignore_chars=[","], reverse_num=False):
    number_pattern = r"[-+]?[.]?[\d]+(,\d+)*[\.]?\d*(?:[eE][-+]?\d+)?%?"
    num_char_spans = [(m.start(0), m.end(0)) for m in re.finditer(number_pattern, text)]
    if len(num_char_spans) == 0: return text
    out_text = ""
    last_e = -1
    for i, (s, e) in enumerate(num_char_spans):
        out_text += text[:s] if i == 0 else text[last_e:s]
        num_str = delim.join([c for c in list(text[s:e]) if c not in ignore_chars])
        out_text += num_str if not reverse_num else num_str[::-1]
        last_e = e
    out_text += text[last_e:]  # append rest
    return out_text


def random_sample_numbers(with_vars):
    # the number of var_numbers
    op_num = random.randint(1, 2)
    candi_num = 30
    text_mapping = [chr(i) for i in list(range(65, 91)) + list(range(97, 122))]
    shuffle(text_mapping)
    var_numbers = []
    real_numbers = []
    candidate_numbers = []
    for i in range(candi_num):
        # random sample a number
        # 1000 float number
        is_int = random.randint(0, 9) < 8
        if is_int:
            final_num = str(random.randint(1, 100))
        else:
            final_num = str(random.randint(1, 1000) / 10)
        if i <= op_num:
            var_numbers.append(text_mapping[i])
            real_numbers.append(final_num)
            # random sample a + and -
            operator = random.choice(["*", "/"])
            if i != op_num:
                var_numbers.append(operator)
                real_numbers.append(operator)
        if i >= op_num and not with_vars:
            break
        candidate_numbers.append(final_num)
    if with_vars:
        input_expression = " ".join(var_numbers)
        zipped_values = list(zip(text_mapping[:candi_num], candidate_numbers))
        shuffle(zipped_values)
        candi_expression = " ".join(["{} = {} ;".format(var_name, var_value)
                                     for var_name, var_value in zipped_values])
        input_line = input_expression + " col : " + candi_expression
    else:
        input_line = " ".join(real_numbers)
    # always plus 3
    output_num = eval(" ".join(real_numbers)) + 1
    if isinstance(output_num, int):
        output_line = str(output_num)
    else:
        output_line = "{:.1f}".format(eval(" ".join(real_numbers)))
    return input_line, output_line


if __name__ == '__main__':
    output_dir = "pretrain_math"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_src_f = open(os.path.join(output_dir, "train.src"), "w", encoding="utf8")
    train_tgt_f = open(os.path.join(output_dir, "train.tgt"), "w", encoding="utf8")
    dev_src_f = open(os.path.join(output_dir, "dev.src"), "w", encoding="utf8")
    dev_tgt_f = open(os.path.join(output_dir, "dev.tgt"), "w", encoding="utf8")
    for _ in tqdm(range(4000000)):
        input_line, output_line = random_sample_numbers(with_vars=True)
        input_line = expand_numbers_in_text(input_line)
        output_line = expand_numbers_in_text(output_line)
        train_src_f.write(input_line + "\n")
        train_tgt_f.write(output_line + "\n")

    for _ in tqdm(range(20000)):
        input_line, output_line = random_sample_numbers(with_vars=True)
        input_line = expand_numbers_in_text(input_line)
        output_line = expand_numbers_in_text(output_line)
        dev_src_f.write(input_line + "\n")
        dev_tgt_f.write(output_line + "\n")

    train_src_f.close()
    train_tgt_f.close()
    dev_src_f.close()
    dev_tgt_f.close()
