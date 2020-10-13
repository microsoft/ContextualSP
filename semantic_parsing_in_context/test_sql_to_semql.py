import json
from context.converter import SQLConverter, SparcDBContext
import unittest
from allennlp.data.tokenizers import WordTokenizer


class TestSQLToSemQL(unittest.TestCase):

    @staticmethod
    def template(sql_plain, sql_text, db_id, expected_str):
        sql_clause = json.loads(sql_text)
        db_context = SparcDBContext(db_id=db_id,
                                    utterance=[],
                                    tokenizer=WordTokenizer(),
                                    # TODO: Please first config the dataset path you want to test
                                    tables_file="dataset_sparc\\tables.json",
                                    database_path="dataset_sparc\\database")
        converter = SQLConverter(db_context=db_context)
        inter_seq = converter.translate_to_intermediate(sql_clause=sql_clause)
        assert str(inter_seq) == expected_str, \
            f'\nSQL:\t\t{sql_plain}\nExp:\t\t{expected_str}\nPred:\t\t{str(inter_seq)}\n'

    def test_example(self):
        db_id = "flight_2"
        sql_plain = "SELECT * FROM AIRLINES"
        sql_clause = """
        {
            "orderBy": [], 
            "from": {
                "table_units": [
                    [
                        "table_unit", 
                        0
                    ]
                ], 
                "conds": []
            }, 
            "union": null, 
            "except": null, 
            "groupBy": [], 
            "limit": null, 
            "intersect": null, 
            "where": [], 
            "having": [], 
            "select": [
                false, 
                [
                    [
                        0, 
                        [
                            0, 
                            [
                                0, 
                                0, 
                                false
                            ], 
                            null
                        ]
                    ]
                ]
            ]
        }
        """
        expected_action_str = "[Statement -> Root, Root -> Select, Select -> A, A -> none C T, C -> *, T -> airlines]"
        self.template(sql_plain, sql_clause, db_id, expected_action_str)
