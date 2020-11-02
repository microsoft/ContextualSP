# coding: utf-8

import json

all_examples = {
    'trian': json.load(open('data/spider/train_spider.json', 'r', encoding='utf-8')),
    'dev': json.load(open('data/spider/dev.json', 'r', encoding='utf-8'))
}


def search_for_id(question, split='dev'):
    examples = all_examples[split]
    for idx, example in enumerate(examples):
        if example['question'] == question:
            return idx


if __name__ == '__main__':
    question = 'Find the last name of the student who has a cat that is age 3.'
    print(search_for_id(question))