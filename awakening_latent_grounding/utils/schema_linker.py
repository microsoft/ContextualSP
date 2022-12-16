import numpy as np
import torch
from utils.data_types import *
from utils.nlp_utils import *

class GreedyLinker:
    schema: SpiderSchema
    question: Utterance
    matched_values: List[ValueMatch]
    threshold: float
    identify_results: Dict[SQLTokenType, List[float]]
    alignment_dict: Dict[Tuple[SQLTokenType, int, int], Tuple[float, int]]

    def __init__(self, schema: SpiderSchema, question: Utterance, matched_values: List[ValueMatch], threshold=0.3) -> None:
        self.schema = schema
        self.question = question
        self.matched_values = matched_values
        self.threshold = threshold
        pass
    
    def link(self, identify_results: Dict[SQLTokenType, List[float]], align_weights: Dict[SQLTokenType, List[List[float]]]) -> List[AlignmentLabel]:
        self.identify_results = identify_results

        alignments = self.init_skeleton_alignments(align_weights)
        alignments = self.fix_missing(alignments)
        return alignments
    
    def fix_missing(self, alignments: List[AlignmentLabel]) -> List[AlignmentLabel]:
        # fix column suffix missing
        for i, align_label in enumerate(alignments):
            if align_label.align_type != SQLTokenType.column:
                continue
            column_original_lemma = self.schema.column_names_lemma[self.schema.id_map[align_label.align_value]].split()
            if i + 1 < len(alignments) and alignments[i+1].align_type == SQLTokenType.null and alignments[i+1].token.lemma in column_original_lemma:
                alignments[i+1] = AlignmentLabel(token=alignments[i+1].token, align_type=SQLTokenType.column, align_value=align_label.align_value, confidence=align_label.confidence)

        # fix column prefix missing
        for i in range(len(alignments)-1, -1, -1):
            align_label = alignments[i]
            if align_label.align_type != SQLTokenType.column:
                continue

            column_original_lemma = self.schema.column_names_lemma[self.schema.id_map[align_label.align_value]].split()
            if i - 1 >= 0 and alignments[i-1].align_type == SQLTokenType.null and alignments[i-1].token.lemma in column_original_lemma:
                alignments[i-1] = AlignmentLabel(token=alignments[i-1].token, align_type=SQLTokenType.column, align_value=align_label.align_value, confidence=align_label.confidence)


        schema_extact_matches = self._lookup_all_schema_extract_matches()

        # Fix column missing with value exists
        for i, align_label in enumerate(alignments):
            if align_label.align_type != SQLTokenType.value:
                continue
            
            column_name = align_label.align_value.replace("VAL_", "")
            column_idx = self.schema.id_map[column_name]
            for match in schema_extact_matches:
                if match['type'] != SQLTokenType.column or match['id'] != column_idx or not match['is_distinct']:
                    continue
                
                is_all_unmatched = True
                for q_idx in range(match['start'], match['end'] + 1):
                    if alignments[q_idx].align_type != SQLTokenType.null:
                        is_all_unmatched = False
                        break
                
                if is_all_unmatched:
                    for q_idx in range(match['start'], match['end'] + 1):
                        alignments[q_idx] = AlignmentLabel(self.question.tokens[q_idx], align_type=SQLTokenType.column, align_value=column_name, confidence=1.0)

        # Fix table column that occurs multiple times
        for match in schema_extact_matches:
            if self.identify_results[match['type']][match['id']] < 0.5 or not match['is_distinct']:
                continue

            is_all_unmatched = True
            for q_idx in range(match['start'], match['end'] + 1):
                if alignments[q_idx].align_type != SQLTokenType.null:
                    is_all_unmatched = False
                    break
            
            if is_all_unmatched:
                align_value = self.schema.get_identifier_name(type=match['type'].abbr, index=match['id'])
                for q_idx in range(match['start'], match['end'] + 1):
                    alignments[q_idx] = AlignmentLabel(self.question.tokens[q_idx], align_type=match['type'], align_value=align_value, confidence=1.0)

        return alignments
    
    def _lookup_extract_tokens(self, query: str) -> List[Tuple[int, int]]:
        ngrams = permutate_ngrams(tokens=[x.lemma for x in self.question.tokens])
        matched_spans = []
        for i, j, ngram in ngrams:
            if ngram == query:
                matched_spans.append((i, j))
        return matched_spans
    
    def _lookup_all_schema_extract_matches(self, identify_threshold: float=0.5) -> List[Dict]:
        schema_matches = []
        for tbl_idx in range(self.schema.num_tables):
            if self.identify_results[SQLTokenType.table][tbl_idx] < identify_threshold:
                continue

            table_lemma = self.schema.table_names_lemma[tbl_idx]
            for start, end in self._lookup_extract_tokens(table_lemma):
                match = { 'type': SQLTokenType.table, 'id': tbl_idx, 'start': start, 'end': end, 'is_distinct': True }
                schema_matches.append(match)
        
        for col_idx in range(self.schema.num_columns):
            if self.identify_results[SQLTokenType.column][col_idx] < identify_threshold:
                continue

            column_lemma = self.schema.column_names_lemma[col_idx]
            for start, end in self._lookup_extract_tokens(column_lemma):
                match = { 'type': SQLTokenType.column, 'id': col_idx, 'start': start, 'end': end, 'is_distinct': True }
                schema_matches.append(match)

        for i, match in enumerate(schema_matches):
            is_distinct = match['is_distinct']
            for j in range(i + 1, len(schema_matches)):
                match_j = schema_matches[j]
                if match_j['start'] > match['end'] or match_j['end'] < match['start']:
                    continue
                is_distinct = False
                match_j['is_distinct'] = False
            match['is_distinct'] = is_distinct
        
        return schema_matches

    def _is_ngram_tokens(self, token1: Token, token2: Token, entity: str):
        ngram12 = '{} {}'.format(token1.lemma, token2.lemma)
        return ngram12 in entity
    
    def init_skeleton_alignments(self, align_weights: Dict[SQLTokenType, List[List[float]]]):
        alignments = self._init_alignments(align_weights)

        question_align_labels = []
        for q_idx in range(len(self.question.tokens)):
            if q_idx not in alignments or len(alignments) == 0:
                question_align_labels.append(AlignmentLabel(token=self.question.tokens[q_idx], align_type=SQLTokenType.null, align_value=None, confidence=1.0))
                continue

            question_align_labels.append(alignments[q_idx][0])

        return question_align_labels

    def _init_alignments(self, align_weights: Dict[SQLTokenType, List[List[float]]]) -> Dict[int, List[AlignmentLabel]]:
        alignments = defaultdict(list)
        threshold = self.threshold
        low_threshold = max(0.2, 1.0 / self.question.num_tokens)
        
        # First is Value as value has span information
        val_align_weights = align_weights[SQLTokenType.value]
        col_align_weights = align_weights[SQLTokenType.column]
        tbl_align_weights = align_weights[SQLTokenType.table]

        columns_with_value = set([])
        for v_idx, value in enumerate(self.matched_values):
            if self.identify_results[SQLTokenType.value][v_idx] < 0.5:
                continue
            confidence = self.identify_results[SQLTokenType.value][v_idx]
            for q_idx in range(value.start, value.end + 1):
                alignments[q_idx].append(AlignmentLabel(
                    token=self.question.tokens[q_idx],
                    align_type=SQLTokenType.value,
                    align_value="VAL_{}".format(value.column),
                    confidence=confidence))

            columns_with_value.add(value.column)
        
        for c_idx in range(1, self.schema.num_columns): # Ignore column *
            if self.identify_results[SQLTokenType.column][c_idx] < 0.5:
                continue

            align_vector = np.array(col_align_weights[c_idx]) # * self.identify_results[SQLTokenType.column][c_idx]
            ranks = np.argsort(align_vector)[::-1]
            total_score = 0.0
            for rk in range(len(align_vector)):
                q_idx = ranks[rk]
                score = align_vector[q_idx]
                if score < threshold / len(self.schema.column_names_lemma[c_idx].split()):
                    break
                
                if total_score >= threshold:
                    break
                
                total_score += score
                alignments[q_idx].append(AlignmentLabel(
                            token=self.question.tokens[q_idx],
                            align_type=SQLTokenType.column,
                            align_value=self.schema.get_col_identifier_name(c_idx),
                            confidence=score))

        for t_idx in range(self.schema.num_tables): # Ignore column *
            if self.identify_results[SQLTokenType.table][t_idx] < 0.5:
                continue

            align_vector = np.array(tbl_align_weights[t_idx]) #* self.identify_results[SQLTokenType.table][t_idx]
            ranks = np.argsort(align_vector)[::-1]
            total_score = 0.0
            for rk in range(len(align_vector)):
                q_idx = ranks[rk]
                score = align_vector[q_idx]
                if score < low_threshold or rk > 4:
                    break
                
                if total_score >= threshold:
                    break
                
                total_score += score
                alignments[q_idx].append(AlignmentLabel(
                            token=self.question.tokens[q_idx],
                            align_type=SQLTokenType.table,
                            align_value=self.schema.get_tbl_identifier_name(t_idx),
                            confidence=score
                        ))
        
        for q_idx in alignments:
            alignments[q_idx] = list(sorted(alignments[q_idx], key=lambda x: self.get_alignment_label_sort_weight(x), reverse=True))
        
        return alignments
    
    def get_alignment_label_sort_weight(self, align_label: AlignmentLabel) -> float:
        if align_label.align_type == SQLTokenType.value:
            return 100.0 + align_label.confidence
        
        elif align_label.align_type == SQLTokenType.column:
            column_idx = self.schema.id_map[align_label.align_value]
            weight = 1.0
            if align_label.token.lemma.lower() in self.schema.column_names_original[column_idx].lower():
                weight = 1.5
            return align_label.confidence * weight
        
        elif align_label.align_type == SQLTokenType.table:
            table_idx = self.schema.id_map[align_label.align_value]
            weight = 1.0
            if align_label.token.lemma.lower() in self.schema.table_names_original[table_idx].lower():
                weight *= 1.5
            return align_label.confidence * weight
        else:
            print(align_label)
            raise NotImplementedError()
    
class SpiderGreedyLinker:
    schema: SpiderSchema
    question: Utterance
    matched_values: List[ValueMatch]
    identify_results: Dict[SQLTokenType, List[float]]
    alignment_dict: Dict[Tuple[SQLTokenType, int, int], Tuple[float, int]]
    threshold: float
    def __init__(self, schema: SpiderSchema, question: Utterance, matched_values: List[ValueMatch], threshold=0.3) -> None:
        self.schema = schema
        self.question = question
        self.matched_values = matched_values
        self.threshold = threshold

    '''
    Lookup all linking relations with different confidences
    '''
    def search_all(self, identify_results: Dict[SQLTokenType, List[float]], align_weights: Dict[SQLTokenType, List[List[float]]]) -> List[List[Dict]]:
        assert len(identify_results[SQLTokenType.table]) == len(self.schema.table_names_original)
        assert len(identify_results[SQLTokenType.column]) == len(self.schema.column_names_original)
        assert len(identify_results[SQLTokenType.value]) == len(self.matched_values)
        self.identify_results = identify_results
        
        init_alignments = self._init_alignments(align_weights)
        alignments_with_scores = defaultdict(list)
        for q_idx, align_labels in init_alignments.items():
            for rk, align_label in enumerate(align_labels):
                slsql_label = align_label.to_slsql(self.schema)
                if align_label.confidence > self.threshold and rk < 1:
                    slsql_label['confidence'] = 'high'
                else:
                    slsql_label['confidence'] = 'low'
                
                is_added = False
                for label in alignments_with_scores[q_idx]:
                    if label['type'] == slsql_label['type'] and label['id'] == slsql_label['id']:
                        is_added = True
                        break
                if not is_added:
                    alignments_with_scores[q_idx].append(slsql_label)
            pass

        schema_extact_matches = self._lookup_all_schema_extract_matches()

        for match in schema_extact_matches:
            if self.identify_results[match['type']][match['id']] < 0.5:
                continue
            
            align_value = self.schema.get_identifier_name(match['type'].abbr, match['id'])
            slsql_label = { 'type': match['type'].abbr, 'id': match['id'], 'confidence': 'low', 'value': align_value }
            is_all_unmatched = True
            for q_idx in range(match['start'], match['end'] + 1):
                if q_idx in init_alignments and \
                    not (len(init_alignments[q_idx]) == 1 and init_alignments[q_idx][0].align_type == match['type'] and self.schema.id_map[init_alignments[q_idx][0].align_value] == match['id']):
                    is_all_unmatched = False
                    break

            if is_all_unmatched and match['is_distinct']:
                slsql_label['confidence'] = 'high'

            for q_idx in range(match['start'], match['end'] + 1):
                is_added = False
                for label in alignments_with_scores[q_idx]:
                    if label['type'] == slsql_label['type'] and label['id'] == slsql_label['id']:
                        is_added = True
                        break
                if not is_added:
                    slsql_label['token'] = self.question.tokens[q_idx].token
                    alignments_with_scores[q_idx].append(slsql_label)

        all_alignment_labels = []
        for q_idx in range(self.question.num_tokens):
            if q_idx not in alignments_with_scores:
                all_alignment_labels.append(None)
            else:
                sorted_alignments = sorted(alignments_with_scores[q_idx], key=lambda x: x['confidence'] == 'high', reverse=True)
                alignment_sets = set([])
                distinct_labels = []
                for alignment in sorted_alignments:
                    if (alignment['type'], alignment['id']) in alignment_sets:
                        continue
                    alignment_sets.add((alignment['type'], alignment['id']))
                    distinct_labels.append(alignment)
                all_alignment_labels.append(distinct_labels)

        return all_alignment_labels

    def link(self, identify_results: Dict[SQLTokenType, List[float]], align_weights: Dict[SQLTokenType, List[List[float]]]) -> List[AlignmentLabel]:
        assert len(identify_results[SQLTokenType.table]) == len(self.schema.table_names_original)
        assert len(identify_results[SQLTokenType.column]) == len(self.schema.column_names_original)
        assert len(identify_results[SQLTokenType.value]) == len(self.matched_values)
        self.identify_results = identify_results

        alignments = self.init_skeleton_alignments(align_weights)
        alignments = self.fix_missing(alignments)
        return alignments
    
    def fix_missing(self, alignments: List[AlignmentLabel]) -> List[AlignmentLabel]:
        # fix column suffix missing
        for i, align_label in enumerate(alignments):
            if align_label.align_type != SQLTokenType.column:
                continue
            column_original_lemma = self.schema.column_names_lemma[self.schema.id_map[align_label.align_value]].split()
            if i + 1 < len(alignments) and alignments[i+1].align_type == SQLTokenType.null and alignments[i+1].token.lemma in column_original_lemma:
                alignments[i+1] = AlignmentLabel(token=alignments[i+1].token, align_type=SQLTokenType.column, align_value=align_label.align_value, confidence=align_label.confidence)

        # fix column prefix missing
        for i in range(len(alignments)-1, -1, -1):
            align_label = alignments[i]
            if align_label.align_type != SQLTokenType.column:
                continue

            column_original_lemma = self.schema.column_names_lemma[self.schema.id_map[align_label.align_value]].split()
            if i - 1 >= 0 and alignments[i-1].align_type == SQLTokenType.null and alignments[i-1].token.lemma in column_original_lemma:
                alignments[i-1] = AlignmentLabel(token=alignments[i-1].token, align_type=SQLTokenType.column, align_value=align_label.align_value, confidence=align_label.confidence)


        schema_extact_matches = self._lookup_all_schema_extract_matches()

        # Fix column missing with value exists
        for i, align_label in enumerate(alignments):
            if align_label.align_type != SQLTokenType.value:
                continue
            
            column_name = align_label.align_value.replace("VAL_", "")
            column_idx = self.schema.id_map[column_name]
            for match in schema_extact_matches:
                if match['type'] != SQLTokenType.column or match['id'] != column_idx or not match['is_distinct']:
                    continue
                
                is_all_unmatched = True
                for q_idx in range(match['start'], match['end'] + 1):
                    if alignments[q_idx].align_type != SQLTokenType.null:
                        is_all_unmatched = False
                        break
                
                if is_all_unmatched:
                    for q_idx in range(match['start'], match['end'] + 1):
                        alignments[q_idx] = AlignmentLabel(self.question.tokens[q_idx], align_type=SQLTokenType.column, align_value=column_name, confidence=1.0)

        # Fix table column that occurs multiple times
        for match in schema_extact_matches:
            if self.identify_results[match['type']][match['id']] < 0.5 or not match['is_distinct']:
                continue

            is_all_unmatched = True
            for q_idx in range(match['start'], match['end'] + 1):
                if alignments[q_idx].align_type != SQLTokenType.null:
                    is_all_unmatched = False
                    break
            
            if is_all_unmatched:
                align_value = self.schema.get_identifier_name(type=match['type'].abbr, index=match['id'])
                for q_idx in range(match['start'], match['end'] + 1):
                    alignments[q_idx] = AlignmentLabel(self.question.tokens[q_idx], align_type=match['type'], align_value=align_value, confidence=1.0)

        return alignments
    
    def _lookup_extract_tokens(self, query: str) -> List[Tuple[int, int]]:
        ngrams = permutate_ngrams(tokens=[x.lemma for x in self.question.tokens])
        matched_spans = []
        for i, j, ngram in ngrams:
            if ngram == query:
                matched_spans.append((i, j))
        return matched_spans
    
    def _lookup_all_schema_extract_matches(self, identify_threshold: float=0.5) -> List[Dict]:
        schema_matches = []
        for tbl_idx in range(self.schema.num_tables):
            if self.identify_results[SQLTokenType.table][tbl_idx] < identify_threshold:
                continue

            table_lemma = self.schema.table_names_lemma[tbl_idx]
            for start, end in self._lookup_extract_tokens(table_lemma):
                match = { 'type': SQLTokenType.table, 'id': tbl_idx, 'start': start, 'end': end, 'is_distinct': True }
                schema_matches.append(match)
        
        for col_idx in range(self.schema.num_columns):
            if self.identify_results[SQLTokenType.column][col_idx] < identify_threshold:
                continue

            column_lemma = self.schema.column_names_lemma[col_idx]
            for start, end in self._lookup_extract_tokens(column_lemma):
                match = { 'type': SQLTokenType.column, 'id': col_idx, 'start': start, 'end': end, 'is_distinct': True }
                schema_matches.append(match)

        for i, match in enumerate(schema_matches):
            is_distinct = match['is_distinct']
            for j in range(i + 1, len(schema_matches)):
                match_j = schema_matches[j]
                if match_j['start'] > match['end'] or match_j['end'] < match['start']:
                    continue
                is_distinct = False
                match_j['is_distinct'] = False
            match['is_distinct'] = is_distinct
        
        return schema_matches

    def _is_ngram_tokens(self, token1: Token, token2: Token, entity: str):
        ngram12 = '{} {}'.format(token1.lemma, token2.lemma)
        return ngram12 in entity

    def init_skeleton_alignments(self, align_weights: Dict[SQLTokenType, List[List[float]]]):
        alignments = self._init_alignments(align_weights)

        question_align_labels = []
        for q_idx in range(len(self.question.tokens)):
            if q_idx not in alignments or len(alignments) == 0:
                question_align_labels.append(AlignmentLabel(token=self.question.tokens[q_idx], align_type=SQLTokenType.null, align_value=None, confidence=1.0))
                continue

            question_align_labels.append(alignments[q_idx][0])

        return question_align_labels
    
    def _init_alignments(self, align_weights: Dict[SQLTokenType, List[List[float]]]) -> Dict[int, List[AlignmentLabel]]:
        alignments = defaultdict(list)
        threshold = self.threshold
        low_threshold = max(0.05, 1.0 / self.question.num_tokens)
        
        # First is Value as value has span information
        val_align_weights = align_weights[SQLTokenType.value]
        col_align_weights = align_weights[SQLTokenType.column]
        tbl_align_weights = align_weights[SQLTokenType.table]

        columns_with_value = set([])
        for v_idx, value in enumerate(self.matched_values):
            if self.identify_results[SQLTokenType.value][v_idx] < 0.5:
                continue
            confidence = self.identify_results[SQLTokenType.value][v_idx]
            for q_idx in range(value.start, value.end + 1):
                alignments[q_idx].append(AlignmentLabel(
                    token=self.question.tokens[q_idx],
                    align_type=SQLTokenType.value,
                    align_value="VAL_{}".format(value.column),
                    confidence=confidence))

            columns_with_value.add(value.column)
        
        for c_idx in range(1, self.schema.num_columns): # Ignore column *
            # if self.identify_results[SQLTokenType.column][c_idx] < 0.5:
            #     continue

            align_vector = np.array(col_align_weights[c_idx]) * self.identify_results[SQLTokenType.column][c_idx]
            ranks = np.argsort(align_vector)[::-1]
            for rk in range(len(align_vector)):
                q_idx = ranks[rk]
                score = align_vector[q_idx] #* self.identify_results[SQLTokenType.column][c_idx]
                # if self.schema.get_column_full_name(c_idx) in columns_with_value:
                #     score *= 0.5
                
                if score >= threshold:
                    alignments[q_idx].append(AlignmentLabel(
                        token=self.question.tokens[q_idx],
                        align_type=SQLTokenType.column,
                        align_value=self.schema.get_col_identifier_name(c_idx),
                        confidence=score
                    ))
                
                if score >= low_threshold and self.identify_results[SQLTokenType.column][c_idx] > 0.5:
                    if rk < 1 or self.question.tokens[q_idx].lemma in self.schema.column_names_lemma[c_idx].split(' '):
                        alignments[q_idx].append(AlignmentLabel(
                            token=self.question.tokens[q_idx],
                            align_type=SQLTokenType.column,
                            align_value=self.schema.get_col_identifier_name(c_idx),
                            confidence=score
                        ))

        for t_idx in range(self.schema.num_tables):
            # if self.identify_results[SQLTokenType.table][t_idx] < 0.5:
            #     continue

            align_vector = np.array(tbl_align_weights[t_idx]) * self.identify_results[SQLTokenType.table][t_idx]
            ranks = np.argsort(align_vector)[::-1]
            for rk in range(len(align_vector)):
                q_idx = ranks[rk]
                score = align_vector[q_idx] #* self.identify_results[SQLTokenType.table][t_idx]

                if score >= threshold:
                    alignments[q_idx].append(AlignmentLabel(
                        token=self.question.tokens[q_idx],
                        align_type=SQLTokenType.table,
                        align_value=self.schema.table_names_original[t_idx].lower(),
                        confidence=score
                    ))
                
                if score >= low_threshold and self.identify_results[SQLTokenType.table][t_idx] > 0.5:
                    if rk < 1 or self.question.tokens[q_idx].lemma in self.schema.table_names_lemma[t_idx].split(' '):
                        alignments[q_idx].append(AlignmentLabel(
                            token=self.question.tokens[q_idx],
                            align_type=SQLTokenType.table,
                            align_value=self.schema.get_tbl_identifier_name(t_idx),
                            confidence=score
                        ))
        
        for q_idx in alignments:
            alignments[q_idx] = list(sorted(alignments[q_idx], key=lambda x: self.get_alignment_label_sort_weight(x), reverse=True))

        return alignments

    def get_alignment_label_sort_weight(self, align_label: AlignmentLabel) -> float:
        if align_label.align_type == SQLTokenType.value:
            return 100.0 + align_label.confidence
        
        elif align_label.align_type == SQLTokenType.column:
            column_idx = self.schema.id_map[align_label.align_value]
            weight = 1.0
            if align_label.token.lemma.lower() in self.schema.column_names_original[column_idx].lower():
                weight = 1.5
            return align_label.confidence * weight
        
        elif align_label.align_type == SQLTokenType.table:
            table_idx = self.schema.id_map[align_label.align_value]
            weight = 1.0
            if align_label.token.lemma.lower() in self.schema.table_names_original[table_idx].lower():
                weight *= 1.5
            return align_label.confidence * weight
        else:
            print(align_label)
            raise NotImplementedError()


def greedy_link_spider(
    identify_logits: Dict[SQLTokenType, torch.Tensor],
    alignment_weights: Dict[SQLTokenType, torch.Tensor],
    question: Utterance,
    schema: SpiderSchema,
    values: List[ValueMatch],
    threshold: float = 0.25
) -> List[AlignmentLabel]:

    linker = SpiderGreedyLinker(schema=schema, question=question, matched_values=values, threshold=threshold)
    for align_type in identify_logits:
        identify_logits[align_type] = torch.softmax(identify_logits[align_type], dim=-1)[:, 1].cpu().tolist()
        alignment_weights[align_type] = alignment_weights[align_type].cpu().tolist()

    alignment_labels = linker.link(identify_results=identify_logits, align_weights=alignment_weights)
    return alignment_labels


def greedy_search_all_spider(
    identify_logits: Dict[SQLTokenType, torch.Tensor],
    alignment_weights: Dict[SQLTokenType, torch.Tensor],
    question: Utterance,
    schema: SpiderSchema,
    values: List[ValueMatch],
    threshold: float = 0.3
) -> List[List[Dict]]:

    linker = SpiderGreedyLinker(schema=schema, question=question, matched_values=values, threshold=threshold)
    return linker.search_all(identify_results=identify_logits, align_weights=alignment_weights)

def mask_value_alignments(align_weights: torch.Tensor, values: List[ValueMatch]) -> torch.Tensor:
    for v_i in range(len(align_weights)):
        start, end = values[v_i].start, values[v_i].end
        mask = torch.zeros(align_weights.size(1), dtype=torch.bool)
        mask[start:end+1] = 1
        align_weights[v_i].masked_fill_(mask == 0, 0.0)
    return align_weights

def generate_alignments_spider(
    align_weights: Dict[SQLTokenType, torch.Tensor],
    question: Utterance,
    schema: SpiderSchema,
    values: List[ValueMatch],
    threshold: float=0.3,
) -> List[AlignmentLabel]:
    assert len(align_weights[SQLTokenType.table]) == schema.num_tables
    assert len(align_weights[SQLTokenType.column]) == schema.num_columns
    assert len(align_weights[SQLTokenType.value]) == len(values)

    align_weights[SQLTokenType.value] = mask_value_alignments(align_weights[SQLTokenType.value], values)

    align_matrix = torch.cat([align_weights[SQLTokenType.table], align_weights[SQLTokenType.column], align_weights[SQLTokenType.value]], dim=0)
    align_matrix = align_matrix.transpose(0, 1) # question_length * num_entities
    assert len(align_matrix) == question.num_tokens
    
    align_labels = []
    for q_idx in range(question.num_tokens):
        max_idx = torch.argmax(align_matrix[q_idx], dim=-1).item()
        confidence = align_matrix[q_idx, max_idx]
        if confidence < threshold:
            align_label = AlignmentLabel(question.tokens[q_idx], SQLTokenType.null, None, 1 - confidence)
            align_labels.append(align_label)
            continue

        if max_idx < schema.num_tables:
            align_labels.append(AlignmentLabel(question.tokens[q_idx], SQLTokenType.table, schema.get_tbl_identifier_name(max_idx), confidence))
        elif max_idx < schema.num_tables + schema.num_columns:
            column_idx = max_idx - schema.num_tables
            align_labels.append(AlignmentLabel(question.tokens[q_idx], SQLTokenType.column, schema.get_col_identifier_name(column_idx), confidence))
        elif max_idx < schema.num_tables + schema.num_columns + len(values):
            value_idx = max_idx - schema.num_tables - schema.num_columns
            align_labels.append(AlignmentLabel(question.tokens[q_idx], SQLTokenType.value, 'VAL_{}'.format(values[value_idx].column), confidence))
        else:
            raise NotImplementedError()
    
    return align_labels