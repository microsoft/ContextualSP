"""
Based on https://github.com/ElementAI/Unisar/blob/master/Unisar/api.py
"""
import os
import subprocess
from typing import Optional

import torch

from genre.fairseq_model import GENRE
from semparse.contexts.spider_db_context import SpiderDBContext
from semparse.sql.spider import load_original_schemas, load_tables
from semparse.sql.spider_utils import read_dataset_schema
from step1_schema_linking import read_database_schema
from step2_serialization import build_schema_linking_data
from step3_evaluate import decode_with_constrain, get_alias_schema, post_processing_sql


class UnisarAPI(object):
    def __init__(self, logdir: str, config_path: str):
        self.model = self.inferer.load_model(logdir, step=None)


def convert_csv_to_sqlite(csv_path: str):
    # TODO: infer types when importing
    db_path = csv_path + ".sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
    subprocess.run(["sqlite3", db_path, ".mode csv", f".import {csv_path} Data"])
    return db_path


class UnisarAPI(object):
    """Run Unisar model on a given database."""

    def __init__(self, log_dir: str, db_path: str, schema_path: Optional[str], stanza_model):
        self.log_dir = log_dir
        self.db_path = db_path
        self.schema_path = schema_path
        self.stanza_model = stanza_model

        # if self.db_path.endswith(".sqlite"):
        #     pass
        # elif self.db_path.endswith(".csv"):
        #     self.db_path = convert_csv_to_sqlite(self.db_path)
        # else:
        #     raise ValueError("expected either .sqlite or .csv file")

        self.schema = read_dataset_schema(self.schema_path, stanza_model)
        _, _, self.database_schemas = read_database_schema(self.schema_path)
        self.model = GENRE.from_pretrained(self.log_dir).eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def infer_query(self, question, db_id):
        ###step-1 schema-linking
        lemma_utterance_stanza = self.stanza_model(question)
        lemma_utterance = [word.lemma for sent in lemma_utterance_stanza.sentences for word in sent.words]
        db_context = SpiderDBContext(db_id,
                                     lemma_utterance,
                                     tables_file=self.schema_path,
                                     dataset_path=self.db_path,
                                     stanza_model=self.stanza_model,
                                     schemas=self.schema,
                                     original_utterance=question)
        value_match, value_alignment, exact_match, partial_match = db_context.get_db_knowledge_graph(db_id)

        item = {}
        item['interaction'] = [{'db_id': db_id,
                                'question': question,
                                'sql': '',
                                'value_match': value_match,
                                'value_alignment': value_alignment,
                                'exact_match': exact_match,
                                'partial_match': partial_match,
                                }]

        ###step-2 serialization
        source_sequence, _ = build_schema_linking_data(schema=self.database_schemas[db_id],
                                                       question=question,
                                                       item=item,
                                                       turn_id=0,
                                                       linking_type='default')
        slml_question = source_sequence[0]

        ###step-3 prediction
        schemas, eval_foreign_key_maps = load_tables(self.schema_path)
        original_schemas = load_original_schemas(self.schema_path)
        alias_schema = get_alias_schema(schemas)
        rnt = decode_with_constrain(slml_question, alias_schema[db_id], self.model)
        predict_sql = rnt[0]['text'] if isinstance(rnt[0]['text'], str) else rnt[0]['text'][0]
        score = rnt[0]['score'].tolist()

        predict_sql = post_processing_sql(predict_sql, eval_foreign_key_maps[db_id], original_schemas[db_id],schemas[db_id])

        return {
            "slml_question": slml_question,
            "predict_sql": predict_sql,
            "score": score
        }

    def execute(self, query):
        ### TODO: replace the query with value version
        pass
        # conn = sqlite3.connect(self.db_path)
        # # Temporary Hack: makes sure all literals are collated in a case-insensitive way
        # query = add_collate_nocase(query)
        # results = conn.execute(query).fetchall()
        # conn.close()
        # return results
