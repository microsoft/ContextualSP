import os, sys
import json
import numpy as np
import re
import inflect
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from tqdm import tqdm
sys.path.append('../')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start_index', help='Path to load verifier model')
parser.add_argument('--num_examples', help='local rank')
parser.add_argument('--local_rank')
args = parser.parse_args()

def write(d, f):
    json.dump(d, f)
    f.write('\n')


inflect = inflect.engine()
def check_contain_upper(self, password):
    pattern = re.compile('[A-Z]+')
    match = pattern.findall(password)
    if match:
        return True
    else:
        return False


class SearchQuery():
    # @classmethod
    # def claim2text(cls, claim, type='text'):
    #     search_body = {
    #         "query": {
    #             "match": {
    #                 type: claim
    #             }
    #         }
    #     }
    #     return search_body

    @classmethod
    def claim2text(cls, claim):
        # score in both text and title
        search_body = {
            "query": {
                "multi_match": {
                    "query": claim,
                    "fields": ['text'],
                    "fuzziness": "AUTO"
                }
            }
        }
        return search_body

    @classmethod
    def kws2title(cls, multi_claim):
        search_body = {
            "query": {
                "bool": {
                    "should": [

                    ]
                }
            }}
        for claim in multi_claim:
            tiny_body = {
                "match_phrase": {
                    "title": {
                        'query': claim,
                        "slop": 2
                    }

                    # "slop": 5
                }
            }
            search_body['query']['bool']['should'].append(tiny_body)
        return search_body


class MyElastic():
    def __init__(self, index_name='wiki_search'):
        self.es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])
        self.index_name = index_name
        body = {
            "properties": {
                "id": {
                    "type": "keywords",
                    # "analyzer":"not_analyzed"
                }
            }
        }

        if not self.es.indices.exists(index=self.index_name,request_timeout=60):
            self.es.indices.create(self.index_name,request_timeout=60)
            self.es.indices.put_mapping(index=self.index_name, doc_type='wiki_title',
                                        body=body, include_type_name=True)
            # self.es.indices.put_mapping(index=self.index_name, doc_type='wiki_sentence',
            #                             body=body, include_type_name=True)

    def search(self, search_body):
        ret = self.es.search(index=self.index_name, body=search_body, size=10)
        return ret
    
    def search_by_text(self,query):
        search_body = SearchQuery.claim2text(query)
        ret = self.search(search_body)
        return ret



if __name__ == '__main__':
    ES = MyElastic()
    start_index = int(args.start_index)
    num_examples = int(args.num_examples)
    basic_dir = './LogiGAN'
    with open(f"{basic_dir}/data/gan_corpus_new/beta/ver_train.jsonl", "r") as fr:
        with open(f"{basic_dir}/data/gan_corpus_new/es/ver_train_{start_index}_{start_index + num_examples}.jsonl", "w") as fw:
            cnt = 0
            for l in fr.readlines():
                if cnt < start_index: 
                    cnt += 1
                    continue
                print(f"From local rank {args.local_rank}: {num_examples} left.")
                if num_examples == 0: break
                dic = json.loads(l)
                in_ = dic["input"]
                out_ = dic["output"]
                res = ES.search_by_text(out_)["hits"]["hits"]
                profounds = [r["_source"]["text"] for r in res[1:]]
                for p in profounds:
                    d = {"input": in_, "conclusion": p, "is_gold": 0}
                    write(d, fw)
                d = {"input": in_, "conclusion": out_, "is_gold": 1}
                write(d, fw)
                num_examples -= 1