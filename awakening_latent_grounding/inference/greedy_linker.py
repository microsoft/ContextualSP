"""
A greedy linker to generate binding sequence based on model prediction results
"""
from collections import defaultdict
from typing import List

from .bind_types import (
    LanguageCode,
    NLBindingType,
    NLToken, NLColumn,
    NLBindingTermResult,
    NLBindingRequest,
    NLBindingToken
)

def greedy_link(request: NLBindingRequest, term_results: List[NLBindingTermResult], threshold: float) -> List[NLBindingToken]:
    skeleton_bindings = _init_skeleton_groundings(request, term_results, threshold)
    grounding_tokens = _fix_groundings(list(skeleton_bindings), term_results, request.columns)

    return grounding_tokens

def _contains_token(src_tokens: List[NLToken], tgt_token: NLToken):
    for src_token in src_tokens:
        if src_token.lemma == tgt_token.lemma:
            return True
    return False

def _fix_groundings(grounding_tokens: List[NLBindingToken], term_results: List[NLBindingTermResult], columns: List[NLColumn]) -> List[NLBindingToken]:
    def _get_grounding_score(token: NLBindingToken, index: int):
        for term_result in term_results:
            if term_result.term_type == token.term_type and term_result.term_value == token.term_value:
                return term_result.grounding_scores[index]
        return 0.0

    def _try_fix_missing(base_idx: int, cur_idx: int) -> bool:
        base_token = grounding_tokens[base_idx]
        cur_token = grounding_tokens[cur_idx]
        if base_token.term_type == NLBindingType.Null or cur_token.term_type != NLBindingType.Null:
            return False

        add_missing = False
        cur_score = _get_grounding_score(base_token, cur_idx)

        if base_token.term_type == NLBindingType.Column:
            add_missing = _contains_token(columns[base_token.term_index].tokens, cur_token) and cur_score >= max(0.05, 1.0 / len(grounding_tokens))
        else:
            add_missing = cur_score >= 0.2 and abs(base_idx - cur_idx) <= 1

        if add_missing:
            new_token = NLBindingToken(
                token=cur_token.token,
                lemma=cur_token.lemma,
                term_type=base_token.term_type,
                term_index=base_token.term_index,
                term_value=base_token.term_value,
                confidence=cur_score
            )

            grounding_tokens[cur_idx] = new_token

        return add_missing

    index = 0
    while index < len(grounding_tokens):
        next_idx = index + 1
        while next_idx < len(grounding_tokens) and _try_fix_missing(index, next_idx):
            next_idx += 1
        index = next_idx

    index = len(grounding_tokens) - 1
    while index >= 0:
        next_idx = index - 1
        while next_idx >= 0 and _try_fix_missing(index, next_idx):
            next_idx -= 1
        index = next_idx

    return grounding_tokens

def _is_phrase_grounding_score(score):
    return score < -1.0

def convert_phrase_tokens(query_tokens: List[str], grounding_scores: List[float]):
    spm_indices, spm_tokens, idx_mappings = [], [], []
    for idx, token in enumerate(query_tokens):
        if _is_phrase_grounding_score(grounding_scores[idx]):
            spm_tokens[-1].token += token.token
            spm_tokens[-1].lemma += token.lemma
        else:
            spm_tokens.append(token.clone())
            spm_indices.append(idx)

        idx_mappings.append(len(spm_tokens) - 1)
    return spm_tokens, spm_indices, idx_mappings

def postprocess_term_results(lang: LanguageCode, query_tokens: List[NLToken], term_results: List[NLBindingTermResult]):
    if len(term_results) == 0 or not lang.is_char_based():
        return query_tokens, list(range(len(query_tokens)))

    query_spm_tokens, spm_indices, idx_mappings = convert_phrase_tokens(query_tokens, term_results[0].grounding_scores)
    for term_result in term_results:
        term_result.grounding_scores = [term_result.grounding_scores[i] for i in spm_indices]
    return query_spm_tokens, idx_mappings

def _init_skeleton_groundings(request: NLBindingRequest, term_results: List[NLBindingTermResult], threshold: float):
    skeleton_bindings = defaultdict(list)
    tokens, idx_mappings = postprocess_term_results(request.language, request.question_tokens, term_results)
    columns = request.columns

    for term_result in term_results:
        if term_result.term_score < 0.5:
            continue

        if term_result.term_type == NLBindingType.Value:
            value = request.matched_values[term_result.term_index]
            val_start, val_end = idx_mappings[value.start], idx_mappings[value.end]
            for q_idx in range(val_start, val_end + 1):
                val_token = NLBindingToken(
                    token=tokens[q_idx].token,
                    lemma=tokens[q_idx].lemma,
                    term_type=term_result.term_type,
                    term_index=term_result.term_index,
                    term_value=term_result.term_value,
                    confidence=term_result.term_score
                )

                skeleton_bindings[q_idx].append(val_token)

            continue

        grounding_scores = term_result.grounding_scores

        sorted_grounding_score_with_indices = sorted(enumerate(grounding_scores), key=lambda x: x[1], reverse=True)
        for rk, (q_idx, grounding_score) in enumerate(sorted_grounding_score_with_indices):
            is_skeleton = rk < 3 and grounding_score >= max(2.0 / len(grounding_scores), threshold)

            if term_result.term_type == NLBindingType.Column:
                if _contains_token(columns[term_result.term_index].tokens, tokens[q_idx]):
                    is_skeleton = is_skeleton or grounding_score > max(threshold, 0.1)

            if not is_skeleton:
                continue

            grounding_token = NLBindingToken(
                token=tokens[q_idx].token,
                lemma=tokens[q_idx].lemma,
                term_type=term_result.term_type,
                term_index=term_result.term_index,
                term_value=term_result.term_value,
                confidence=term_result.term_score * grounding_score
            )

            skeleton_bindings[q_idx].append(grounding_token)

    for q_idx, q_token in enumerate(tokens):
        if q_idx not in skeleton_bindings:
            yield NLBindingToken(
                token=q_token.token,
                lemma=q_token.lemma,
                term_type=NLBindingType.Null,
                term_index=q_idx,
                term_value=q_token.token,
                confidence=1.0
            )

            continue

        yield list(sorted(skeleton_bindings[q_idx], key=lambda x: x.confidence, reverse=True))[0]
