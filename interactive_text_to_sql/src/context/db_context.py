import re
from collections import Set, defaultdict
from typing import Dict, Tuple, List

from allennlp.data import Tokenizer, Token
from ordered_set import OrderedSet
from unidecode import unidecode

from .utils import TableColumn, read_dataset_schema, read_dataset_values
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph

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


class SparcDBContext:
    db_schemas = {}
    db_schemas_id_col = {}
    db_schemas_id_tab = {}
    db_tables_data = {}

    def __init__(self, db_id: str, tokenizer: Tokenizer, tables_file: str, database_path: str, utterance: List[Token]):
        self.database_path = database_path
        self.tables_file = tables_file
        self.db_id = db_id
        self.tokenized_utterance = utterance

        if db_id not in SparcDBContext.db_schemas:
            SparcDBContext.db_schemas, SparcDBContext.db_schemas_id_col, SparcDBContext.db_schemas_id_tab \
                = read_dataset_schema(self.tables_file)
        self.schema = SparcDBContext.db_schemas[db_id]
        # get id to column/table
        self.id_to_col = SparcDBContext.db_schemas_id_col[db_id]
        self.id_to_tab = SparcDBContext.db_schemas_id_tab[db_id]

        self.knowledge_graph = self.get_db_knowledge_graph(db_id)

        entity_texts = [self.knowledge_graph.entity_text[entity].lower()
                        for entity in self.knowledge_graph.entities]
        entity_tokens = tokenizer.batch_tokenize(entity_texts)
        self.entity_tokens = entity_tokens

    @staticmethod
    def entity_key_for_column(table_name: str, column: TableColumn) -> str:
        if column.foreign_key is not None:
            column_type = "foreign"
        elif column.is_primary_key:
            column_type = "primary"
        else:
            column_type = column.column_type
        return f"column:{column_type.lower()}:{table_name.lower()}:{column.name.lower()}"

    def get_db_knowledge_graph(self, db_id: str) -> KnowledgeGraph:
        entities: Set[str] = set()
        neighbors: Dict[str, OrderedSet[str]] = defaultdict(OrderedSet)
        entity_text: Dict[str, str] = {}
        foreign_keys_to_column: Dict[str, str] = {}

        db_schema = self.schema
        tables = db_schema.values()

        if db_id not in self.db_tables_data:
            self.db_tables_data[db_id] = read_dataset_values(db_id, self.database_path, tables)

        tables_data = self.db_tables_data[db_id]

        string_column_mapping: Dict[str, set] = defaultdict(set)

        for table, table_data in tables_data.items():
            for table_row in table_data:
                # TODO: special case for column *
                if db_schema[table.name].columns[0].name == '*':
                    columns = db_schema[table.name].columns[1:]
                else:
                    columns = db_schema[table.name].columns
                assert len(columns) == len(table_row)
                for column, cell_value in zip(db_schema[table.name].columns, table_row):
                    if column.column_type == 'text' and type(cell_value) is str:
                        cell_value_normalized = self.normalize_string(cell_value)
                        column_key = self.entity_key_for_column(table.name, column)
                        string_column_mapping[cell_value_normalized].add(column_key)

        for table in tables:
            table_key = f"table:{table.name.lower()}"
            entities.add(table_key)
            entity_text[table_key] = table.text

            for column in db_schema[table.name].columns:
                entity_key = self.entity_key_for_column(table.name, column)
                entities.add(entity_key)
                neighbors[entity_key].add(table_key)
                neighbors[table_key].add(entity_key)
                entity_text[entity_key] = column.text

        # dynamic entities of values in question
        # TODO: we should disable the string match entities in train.
        #  Because it will cause the inconsistent between train and test
        # value_entities = self.get_values_from_question(string_column_mapping)
        #
        # for value_repr, column_keys in value_entities:
        #     entities.add(value_repr)
        #     for column_key in column_keys:
        #         neighbors[value_repr].add(column_key)
        #         neighbors[column_key].add(value_repr)
        #     entity_text[value_repr] = value_repr.replace("string:", "").replace("_", " ")

        # loop again after we have gone through all columns to link foreign keys columns
        for table_name in db_schema.keys():
            for column in db_schema[table_name].columns:
                if column.foreign_key is None:
                    continue

                for foreign_key in column.foreign_key:
                    other_column_table, other_column_name = foreign_key.split(':')

                    # must have exactly one by design
                    other_column = [col for col in db_schema[other_column_table].columns
                                    if col.name == other_column_name][0]

                    entity_key = self.entity_key_for_column(table_name, column)
                    other_entity_key = self.entity_key_for_column(other_column_table, other_column)

                    neighbors[entity_key].add(other_entity_key)
                    neighbors[other_entity_key].add(entity_key)

                    foreign_keys_to_column[entity_key] = other_entity_key

        kg = KnowledgeGraph(entities, dict(neighbors), entity_text)
        kg.foreign_keys_to_column = foreign_keys_to_column

        return kg

    @staticmethod
    def _string_in_table(candidate: str,
                         string_column_mapping: Dict[str, set]) -> List[str]:
        """
        Checks if the string occurs in the table, and if it does, returns the names of the columns
        under which it occurs. If it does not, returns an empty list.
        """
        candidate_column_names: List[str] = []
        # First check if the entire candidate occurs as a cell.
        if candidate in string_column_mapping:
            candidate_column_names = string_column_mapping[candidate]
        # If not, check if it is a substring pf any cell value.
        if not candidate_column_names:
            for cell_value, column_names in string_column_mapping.items():
                if candidate in cell_value:
                    candidate_column_names.extend(column_names)
        candidate_column_names = list(set(candidate_column_names))
        return candidate_column_names

    def get_values_from_question(self,
                                 string_column_mapping: Dict[str, set]) -> List[Tuple[str, str]]:
        entity_data = []
        for i, token in enumerate(self.tokenized_utterance):
            token_text = token.text
            if token_text in STOP_WORDS:
                continue
            normalized_token_text = self.normalize_string(token_text)
            if not normalized_token_text:
                continue
            token_columns = self._string_in_table(normalized_token_text, string_column_mapping)
            if token_columns:
                token_type = token_columns[0].split(":")[1]
                entity_data.append({'value': normalized_token_text,
                                    'token_start': i,
                                    'token_end': i + 1,
                                    'token_type': token_type,
                                    'token_in_columns': token_columns})

        # extracted_numbers = self._get_numbers_from_tokens(self.question_tokens)
        # filter out number entities to avoid repetition
        expanded_entities = []
        for entity in self._expand_entities(self.tokenized_utterance, entity_data, string_column_mapping):
            if entity["token_type"] == "text":
                expanded_entities.append((f"string:{entity['value']}", entity['token_in_columns']))
        # return expanded_entities, extracted_numbers  #TODO(shikhar) Handle conjunctions

        return expanded_entities

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
        string = re.sub("[·・]", ".", string)
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
        # canonicalization rules from sempre.
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
                candidate = "%s_%s" % (current_token, next_token_normalized)
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
