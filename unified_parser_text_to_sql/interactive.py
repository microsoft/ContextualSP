import argparse
import stanza
from unisar.api import UnisarAPI


class Interactive(object):

    def __init__(self, Unisar: UnisarAPI):
        self.unisar = Unisar

    def ask_any_question(self, question, db_id):
        results = self.unisar.infer_query(question, db_id)

        print('input:', results['slml_question'])
        print(f'"pred:" {results["predict_sql"]}  ({results["score"]})')
        # try:
        #     results = self.unisar.execute(results['query'])
        #     print(results)
        # except Exception as e:
        #     print(str(e))

    def show_schema(self, db_id):
        for table in self.unisar.schema[db_id].values():
            print("Table", f"{table.name}")
            for column in table.columns:
                print("    Column", f"{column.name}")

    def run(self, db_id):
        self.show_schema(db_id)
        # self.ask_any_question('Tell me the name about organization', db_id)
        while True:
            question = input("Ask a question: ")
            self.ask_any_question(question, db_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",  default='./models/spider_sl')
    parser.add_argument("--db_id", default='student_1')
    parser.add_argument(
        "--db-path", default='./data/spider/database',
        help="The path to the sqlite database or csv file"
    )
    parser.add_argument(
        "--schema-path", default='./data/spider/tables.json',
        help="The path to the tables.json file with human-readable database schema."
    )
    args = parser.parse_args()

    stanza_model = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
    interactive = Interactive(UnisarAPI(args.logdir, args.db_path, args.schema_path, stanza_model))
    interactive.run(args.db_id)
