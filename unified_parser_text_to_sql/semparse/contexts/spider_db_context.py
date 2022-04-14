import re
from collections import Counter, defaultdict
from typing import Dict, Tuple, List

from unidecode import unidecode

from semparse.sql.spider_utils import TableColumn, read_dataset_schema, read_dataset_values

# == stop words that will be omitted by ContextGenerator
STOP_WORDS = {"", "", "all", "being", "-", "over", "through", "yourselves", "its", "before",
              "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do",
              "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him",
              "nor", "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "'s", "our", "after", "most", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"}

digits_list = [str(i) for i in range(10)]

class SpiderDBContext:
    schemas = {}
    db_knowledge_graph = {}
    db_tables_data = {}

    def __init__(self, db_id: str, utterance: str, tables_file: str, dataset_path: str, stanza_model=None, schemas=None,
                 original_utterance=None):
        self.dataset_path = dataset_path
        self.tables_file = tables_file
        self.db_id = db_id
        self.utterance = utterance
        self.tokenized_utterance = utterance
        self.stanza_model = stanza_model
        self.original_utterance = original_utterance if original_utterance is not None else utterance

        if schemas is not None:
            SpiderDBContext.schemas = schemas
        elif db_id not in SpiderDBContext.schemas:
            SpiderDBContext.schemas = read_dataset_schema(self.tables_file, self.stanza_model)

        self.schema = SpiderDBContext.schemas[db_id]

    @staticmethod
    def entity_key_for_column(table_name: str, column: TableColumn) -> str:
        return f"{table_name.lower()}@{column.name.lower()}"
        if column.foreign_key is not None:
            column_type = "foreign"
        elif column.is_primary_key:
            column_type = "primary"
        else:
            column_type = column.column_type
        return f"column:{column_type.lower()}:{table_name.lower()}:{column.name.lower()}"

    def get_db_knowledge_graph(self, db_id: str):
        db_schema = self.schema
        tables = db_schema.values()

        if db_id not in self.db_tables_data:
            self.db_tables_data[db_id] = read_dataset_values(db_id, self.dataset_path, tables)

        tables_data = self.db_tables_data[db_id]
        string_column_mapping: Dict[str, set] = defaultdict(set)

        for table, table_data in tables_data.items():
            for table_row in table_data:
                for column, cell_value in zip(db_schema[table.name].columns, table_row):
                    if column.column_type == 'text' and type(cell_value) is str:
                        cell_value_normalized = self.normalize_string(cell_value)
                        column_key = self.entity_key_for_column(table.name, column)
                        string_column_mapping[cell_value_normalized].add(column_key)
        # for key in string_column_mapping:
        #     string_column_mapping[key]=list(string_column_mapping[key])

        string_entities = self.get_entities_from_question(string_column_mapping)

        value_match=[]
        value_alignment = []
        for item in string_entities:
            value_match+=item['token_in_columns']
            value_alignment.append(item['alignment'])
        value_match = list(set(value_match))

        r_schemas={}
        for table in db_schema:
            r_schemas["{0}".format(db_schema[table].name).lower()] = db_schema[table].lemma.lower()
            for column in db_schema[table].columns:
                r_schemas[f"{db_schema[table].name}@{column.name}".lower()] = column.lemma.strip('is ').lower()

        question_tokens = [t for t in self.tokenized_utterance]
        schema_counter = Counter()

        partial_match = []
        exact_match = []

        for r_k, r_s in r_schemas.items():
            schema_counter[r_s] = 0
            #exact match
            if r_s in self.tokenized_utterance and r_s not in STOP_WORDS:
                schema_counter[r_s] += 2
            #partial_match
            else:
                for tok in r_s.split(' '):
                    if tok in question_tokens and tok not in STOP_WORDS:
                        schema_counter[r_s]+=1
                        continue

                    for ques_tok in question_tokens:
                        if tok in STOP_WORDS or ques_tok in STOP_WORDS or \
                                len(tok)<=3 or len(ques_tok)<=3:
                            continue
                        if ques_tok in tok or tok in ques_tok:
                            schema_counter[r_s] += 1

            if schema_counter[r_s]>=2:
                exact_match.append(r_k)
            elif schema_counter[r_s]==1:
                partial_match.append(r_k)

        return value_match, value_alignment,  exact_match, partial_match


    def _string_in_table(self, candidate: str,
                         string_column_mapping: Dict[str, set]) -> List[str]:
        """
        Checks if the string occurs in the table, and if it does, returns the names of the columns
        under which it occurs. If it does not, returns an empty list.
        """
        candidate_column_names: List[str] = []
        alignment = []
        # First check if the entire candidate occurs as a cell.
        candidate= candidate.strip('-_"\'')
        if candidate in string_column_mapping and candidate not in digits_list:
            candidate_column_names = string_column_mapping[candidate]
            alignment.append((candidate,candidate))
        # # If not, check if it is a substring pf any cell value.
        # if not candidate_column_names:
        #     for cell_value, column_names in string_column_mapping.items():
        #         if candidate in re.split(' |_|:',
        #                                  cell_value) and candidate not in STOP_WORDS and candidate not in digits_list:
        #             candidate_column_names.extend(column_names)
        #             alignment.append((candidate, cell_value))
        candidate_column_names = list(set(candidate_column_names))
        return candidate_column_names, alignment

    def get_entities_from_question(self, string_column_mapping: Dict[str, set]) -> List[Tuple[str, str]]:
        entity_data = []

        for cell_value, column_names in string_column_mapping.items():
            if (cell_value.replace('_', ' ') in ' '.join(self.utterance) or cell_value.replace('_',
                                                                                               ' ') in self.original_utterance) \
                    and len(re.split('_', cell_value)) >= 2:
                entity_data.append({'value': cell_value,
                                    'token_start': 0,
                                    'token_end': 0,
                                    'alignment': [(cell_value, cell_value)],
                                    'token_in_columns': list(set(column_names))})

        for i, token in enumerate(self.tokenized_utterance):
            token_text = token
            if token_text in STOP_WORDS:
                continue
            normalized_token_text = token_text
            # normalized_token_text = self.normalize_string(token_text)
            if not normalized_token_text:
                continue
            token_columns, alignment = self._string_in_table(normalized_token_text, string_column_mapping)
            if token_columns:
                entity_data.append({'value': normalized_token_text,
                                    'token_start': i,
                                    'token_end': i+1,
                                    'alignment': alignment,
                                    'token_in_columns': token_columns})

        return entity_data

    @staticmethod
    def normalize_string(string: str) -> str:
        """
        These are the transformation rules used to normalize cell in column names in Sempre.  See
        ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
        rules here to normalize and canonicalize cells and columns in the same way so that we can
        match them against constants in logical forms appropriately.
        """
        # Normalization rules from Sempre
        # \u201A -> ,
        string = re.sub("‚", ",", string)
        string = re.sub("„", ",,", string)
        string = re.sub("[·・]", "../sql", string)
        string = re.sub("…", "...", string)
        string = re.sub("ˆ", "^", string)
        string = re.sub("˜", "~", string)
        string = re.sub("‹", "<", string)
        string = re.sub("›", ">", string)
        string = re.sub("[‘’´`]", "'", string)
        string = re.sub("[“”«»]", "\"", string)
        string = re.sub("[•†‡²³]", "", string)
        string = re.sub("[‐‑–—−]", "-", string)
        # Oddly, some unicode characters get converted to _ instead of being stripped.  Not really
        # sure how sempre decides what to do with these...  TODO(mattg): can we just get rid of the
        # need for this function somehow?  It's causing a whole lot of headaches.
        string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
        # This is such a mess.  There isn't just a block of unicode that we can strip out, because
        # sometimes sempre just strips diacritics...  We'll try stripping out a few separate
        # blocks, skipping the ones that sempre skips...
        string = re.sub("[\\u0180-\\u0210]", "", string).strip()
        string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
        string = string.replace("\\n", "_")
        string = re.sub("\\s+", " ", string)
        # Canonicalization rules from Sempre.
        string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return unidecode(string.lower())

    def _expand_entities(self, question, entity_data, string_column_mapping: Dict[str, set]):
        new_entities = []
        for entity in entity_data:
            # to ensure the same strings are not used over and over
            if new_entities and entity['token_end'] <= new_entities[-1]['token_end']:
                continue
            current_start = entity['token_start']
            current_end = entity['token_end']
            current_token = entity['value']
            current_token_type = entity['token_type']
            current_token_columns = entity['token_in_columns']

            while current_end < len(question):
                next_token = question[current_end].text
                next_token_normalized = self.normalize_string(next_token)
                if next_token_normalized == "":
                    current_end += 1
                    continue
                candidate = "%s_%s" %(current_token, next_token_normalized)
                candidate_columns = self._string_in_table(candidate, string_column_mapping)
                candidate_columns = list(set(candidate_columns).intersection(current_token_columns))
                if not candidate_columns:
                    break
                candidate_type = candidate_columns[0].split(":")[1]
                if candidate_type != current_token_type:
                    break
                current_end += 1
                current_token = candidate
                current_token_columns = candidate_columns

            new_entities.append({'token_start': current_start,
                                 'token_end': current_end,
                                 'value': current_token,
                                 'token_type': current_token_type,
                                 'token_in_columns': current_token_columns})
        return new_entities
