import numpy as np
import pandas as pd
import json
import re
import string
import os
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--indicator_type')



def read_indicators(indicator_type):
    assert indicator_type in ["reverse", "premise", "conclusion"]
    indicators = []
    with open(f"../data/Indicators/{indicator_type}.txt", "r") as f:
        for line in f.readlines():
            word = line.strip()
            indicators.append(word)
    return indicators

def locate_logic_indicator_sentences(indicators, text):
    rexp = re.compile("|".join([r"(\b{}\b)".format(ind) for ind in indicators]))
    return list(re.finditer(rexp, text)) 

def find_next_period(current_index, text, forward=True):
    if current_index <= 0: return 0
    if current_index >= len(text) - 1: return len(text) - 1
    stride = 1 if forward else -1
    while True:
        current_index = current_index + stride
        if text[current_index] in [".", "!", "?"]:  
            return current_index + 1 # used directly for subscription
        if current_index >= len(text) - 1: return len(text) - 1
        if current_index <= 0: return 0
        

def is_good_conclusion_indicator(indicator, following_words):
    if indicator == "so":
        following_first_word = following_words.strip().split()[0]
        word_list = [following_first_word]
        for word in word_list:
            pos = nlp(word)[0].pos_
            return not pos in ["ADJ", "ADV", "INTJ", "VERB"] # "so bad", "so many", "so loaded" etc.
    return True

def find_next_char(current_index, text): # non punc, non space
    while text[current_index] not in string.ascii_lowercase + string.digits:
        current_index = current_index + 1
    return current_index


if __name__ == "__main__":
    args = parser.parse_args()
    book_start_index = max(args.start, 0)
    book_end_index = None if args.end == -1 or args.end >= len(os.listdir("../data/BookCorpus/epubtxt/")) else args.end
    indicator_type = args.indicator_type
    indicators = read_indicators(indicator_type)
    all_books = os.listdir("../data/BookCorpus/epubtxt/")
    fw = open(f"./bookcorpus_{indicator_type}/mlm_bookcorpus_{book_start_index}_{book_end_index}.jsonl", "w")
    for book in tqdm(all_books[book_start_index:book_end_index]):
        with open(f"../data/BookCorpus/epubtxt/{book}", "r") as f: 
            book =  " ".join(f.read().lower().split())
            occurances = locate_logic_indicator_sentences(indicators, book)
            for span in occurances:
                try:
                    ind_start, ind_end = span.start(), span.end()
                    mask_start = find_next_char(ind_end, book)
                    mask_end = find_next_period(mask_start + 1, book, True)
                    if book[ind_end] in [".", "!", "?", "'", '"'] or book[ind_end + 1] in [".", "!", "?", "'", '"']: continue # indicator on eos
                    if mask_end - mask_start <= 25 or mask_end - mask_start >= 150: # drop overlong or overshort setence
                        continue
                    if not is_good_conclusion_indicator(book[ind_start: ind_end], book[mask_start:mask_end]):
                        continue

                    pre_len = min(600 + np.random.geometric(p=1/25) * 5, 1000)
                    post_len = min(250 + np.random.geometric(p=1/25) * 5, 600)

                    input_start = find_next_period(ind_start - pre_len, book, False)
                    input_end = find_next_period(ind_end + post_len, book, True)

                    input_p1 = book[input_start: mask_start]
                    input_p2 = "[SEP] [MASK] [SEP]"
                    input_p3 = book[mask_end : input_end]
                    input_ = " ".join([input_p1, input_p2, input_p3]).strip()
                    output_ = book[mask_start : mask_end].strip()

        #             print(input_,)
        #             print(output_, "\n")
                    dic = {"input":input_, "output":output_}
                    json.dump(dic, fw)
                    fw.write("\n")
                except:
                    continue
    fw.close()
