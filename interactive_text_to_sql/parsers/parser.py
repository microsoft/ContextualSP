# coding: utf-8

import json

import dill
import hashlib
import os
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive

from parsers.irnet.context.converter import ActionConverter
from parsers.irnet.dataset_reader.spider_reader import SpiderDatasetReader
from parsers.irnet.models.sparc_parser import SparcParser


class Parser:
    def __init__(self, model: torch.nn.Module):
        assert model is not None
        model.eval()
        self.model = model

    def parse(self, example):
        # requirement: 'predict_sql' or 'predict_semql' must in returned dict
        raise NotImplementedError()


class IRNetSpiderParser(Parser):
    def __init__(self, model):
        super().__init__(model)
        self.spider_dataset_reader = SpiderDatasetReader()
        self.sha1 = hashlib.sha1()

    def parse(self, example):
        hash_id = self.hash_dict(example)[:7]
        if os.path.exists(f'cache/spider_instance/{hash_id}.bin'):
            instance = dill.load(open(f'cache/spider_instance/{hash_id}.bin', 'rb'))
        else:
            db_id = example['db_id']
            inter_utter_list = [example['question']]
            sql_list = [example['sql']]
            sql_query_list = [example['query']]
            instance = self.spider_dataset_reader.text_to_instance(
                utter_list=inter_utter_list,
                db_id=db_id,
                sql_list=sql_list,
                sql_query_list=sql_query_list
            )
            dill.dump(instance, open(f'cache/spider_instance/{hash_id}.bin', 'wb'))
        parsed_result = self.parse_instance(instance)
        return parsed_result

    def parse_instance(self, instance: Instance) -> JsonDict:
        # convert predict result into production rule string
        index_to_rule = [production_rule_field.rule
                         for production_rule_field in instance.fields['valid_actions_list'].field_list[0].field_list]

        # Now get result
        results = sanitize(self.model.forward_on_instance(instance))

        rule_repr = [index_to_rule[ind] for ind in results['best_predict']]
        ground_rule_repr = [index_to_rule[ind] for ind in results['ground_truth']]
        db_context = instance.fields['worlds'].field_list[0].metadata.db_context
        action_converter = ActionConverter(db_context)
        predict_sql = action_converter.translate_to_sql(rule_repr)
        ground_sql = action_converter.translate_to_sql(ground_rule_repr)
        dis_results = {'predict': rule_repr,
                       'predict_sql': predict_sql,
                       'ground': ground_rule_repr,
                       'ground_sql': ground_sql,
                       'table_content': results['table_content']}
        return dis_results

    def hash_dict(self, d):
        dict_str = json.dumps(d)
        self.sha1.update(bytes(dict_str, encoding='utf-8'))
        hex = self.sha1.hexdigest()
        return hex

    @staticmethod
    def get_parser():
        dataset_path = 'data/datasets/spider'
        vocab = Vocabulary.from_files('parsers/irnet/checkpoints/v1.0_spider_baseline_model/vocabulary')
        overrides = {
            "dataset_path": dataset_path,
            "train_data_path": "train.json",
            "validation_data_path": "dev.json"
        }
        parser_model = load_archive('parsers/irnet/checkpoints/v1.0_spider_baseline_model/model.tar.gz',
                                    cuda_device=0,
                                    overrides=json.dumps(overrides)).model
        parser_model.sql_metric_util._evaluator.update_dataset_path(dataset_path=dataset_path)
        parser = IRNetSpiderParser(model=parser_model)
        return parser


if __name__ == '__main__':
    parser: Parser = IRNetSpiderParser.get_parser()
