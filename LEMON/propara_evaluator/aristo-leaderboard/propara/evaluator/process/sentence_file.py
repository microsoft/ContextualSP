from collections import defaultdict
from typing import Dict, List, Tuple


def sentences_from_sentences_file(sentences_filename: str) -> Dict[int, List[str]]:
    all_sentences = dict()  # type: Dict[Tuple[int, int], str]
    with open(sentences_filename) as f:
        for line in f:
            process_id_str, sentence_number_str, text = line.strip().split('\t', 3)[:3]

            process_id = int(process_id_str)
            sentence_number = int(sentence_number_str)

            all_sentences[(process_id, sentence_number)] = text

    sentences_by_process = defaultdict(list)  # type: Dict[int, List[str]]
    for key, sentence in sorted(all_sentences.items()):
        process_id, sentence_number = key
        sentences_by_process[process_id].append(sentence)

    return sentences_by_process
